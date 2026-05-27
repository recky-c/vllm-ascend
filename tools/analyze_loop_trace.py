#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze [LOOP-TRACE] logs produced by VLLM_ASCEND_DEBUG_LOOP_TRACE=1.

This tool is intentionally privacy-preserving: it reads only the structural
fingerprints emitted by `_dbg_loop_consistency` (sum / xorhash / min / max /
shape / dtype / rank) and never requires raw token ids or hidden-state
values to be shared. It is meant to triage the
FLASHCOMM1 + DCP>1 + EAGLE3 (num_spec>1) + async-scheduling output-loop bug
without exposing customer prompts.

Two modes:

1. Single-log mode (default):
       python tools/analyze_loop_trace.py path/to/log --tp-size 16

   Reports
       (A) per-iter cross-rank divergence per tag,
       (B) loop signature on `set_prev_sampled.main`,
       (C/D) handoff consistency for the pcp_full scatter tags,
       (E) for the eagle.* tags, which iter is the first one where the
           rank-0 xorhash starts disagreeing with the most-common pattern
           (helps spot the moment KV cache layout drifts).

2. Compare mode:
       python tools/analyze_loop_trace.py path/to/buggy.log \\
              --compare path/to/control.log --tp-size 16

   For each tag present in both runs, reports the first iter where
   rank-0 xorhash differs between the two runs. This is the cleanest way
   to find the exact code point at which FLASHCOMM1-ON diverges from
   FLASHCOMM1-OFF (or any other A/B you want to run).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict

# New format (preferred):
#   [LOOP-TRACE][<tag>] rank=<r> tp_rank=<t> shape=(..) dtype=.. numel=N
#       sum=.. min=.. max=.. xorhash=..        (int tensors)
#       sum=.. mean=.. min=.. max=.. absmean=. (float tensors)
# Old format (kept for backward compat):
#   [LOOP-TRACE][<tag>] rank=<r> tp_rank=<t> shape=(..) dtype=.. first8=[..] sum32=..
HEAD_RE = re.compile(
    r"\[LOOP-TRACE\]\[(?P<tag>[^\]]+)\]\s+"
    r"rank=(?P<rank>\d+)\s+tp_rank=(?P<tp>-?\d+)\s+"
    r"shape=(?P<shape>\([^)]*\))\s+"
    r"dtype=(?P<dtype>\S+)"
)
SUM_RE = re.compile(r"\bsum=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
XOR_RE = re.compile(r"\bxorhash=(\d+)")
SUM32_RE = re.compile(r"\bsum32=(-?\d+)")  # backward compat


def _extract_fingerprint(line: str) -> str | None:
    """Return a string that uniquely fingerprints the tensor on the line.

    We prefer xorhash (ints) > sum32 (legacy) > sum (floats).  The exact
    numeric type does not matter as long as the same tensor produces the
    same string across rank/iter comparisons.
    """
    m = XOR_RE.search(line)
    if m:
        return f"x{m.group(1)}"
    m = SUM32_RE.search(line)
    if m:
        return f"s{m.group(1)}"
    m = SUM_RE.search(line)
    if m:
        return f"f{m.group(1)}"
    return None


def parse(path: str) -> list[dict]:
    recs: list[dict] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = HEAD_RE.search(line)
            if not m:
                continue
            fp = _extract_fingerprint(line)
            if fp is None:
                continue
            recs.append(
                {
                    "tag": m.group("tag"),
                    "rank": int(m.group("rank")),
                    "tp": int(m.group("tp")),
                    "shape": m.group("shape"),
                    "dtype": m.group("dtype"),
                    "fp": fp,
                    "raw": line.rstrip(),
                }
            )
    return recs


def bucket_by_iter(recs: list[dict], tp_size: int) -> list[dict[str, dict[int, str]]]:
    """Group records into iterations.

    We close an iteration once every rank in [0..tp_size) has emitted
    ``set_prev_sampled.main`` (which is fired exactly once per iter at the
    end of token sampling).  Any records before the first such boundary
    are dropped (they belong to startup / dummy_run).
    """
    iters: list[dict[str, dict[int, str]]] = []
    cur: dict[str, dict[int, str]] = defaultdict(dict)
    seen_main_ranks: set[int] = set()
    for r in recs:
        cur[r["tag"]][r["rank"]] = r["fp"]
        if r["tag"] == "set_prev_sampled.main":
            seen_main_ranks.add(r["rank"])
            if len(seen_main_ranks) >= tp_size:
                iters.append({k: dict(v) for k, v in cur.items()})
                cur = defaultdict(dict)
                seen_main_ranks = set()
    return iters


def report_single(iters: list[dict[str, dict[int, str]]]) -> None:
    n = len(iters)
    print(f"Grouped into {n} iters.\n")
    if n == 0:
        return

    # A) Cross-rank divergence.
    print("=== A) Cross-rank divergence (per iter, per tag) ===")
    diverged: Counter[str] = Counter()
    seen_tags: Counter[str] = Counter()
    for it in iters:
        for tag, by_rank in it.items():
            seen_tags[tag] += 1
            if len(set(by_rank.values())) > 1:
                diverged[tag] += 1
    if not diverged:
        print("  [OK] No cross-rank divergence in any iter.")
    else:
        for tag, k in diverged.most_common():
            print(f"  [DIVERGE] {tag:50s} divergent_iters={k}/{seen_tags[tag]}")

    # B) Loop signature on set_prev_sampled.main (rank 0).
    print("\n=== B) Loop signature in set_prev_sampled.main ===")
    rank0_main = [it["set_prev_sampled.main"].get(0)
                  for it in iters if "set_prev_sampled.main" in it]
    rank0_main = [x for x in rank0_main if x is not None]
    if len(rank0_main) < 4:
        print("  not enough iters to judge loop.")
    else:
        found = False
        for k in (12, 10, 8, 6, 4):
            if len(rank0_main) >= k and len(set(rank0_main[-k:])) == 1:
                print(f"  [LOOP] last {k} iters all produced the same fingerprint on rank 0.")
                found = True
                break
        if not found:
            for period in (2, 3, 4, 5):
                tail = rank0_main[-2 * period:]
                if len(tail) == 2 * period and tail[:period] == tail[period:]:
                    print(f"  [LOOP] tail shows period-{period} repetition on rank 0.")
                    found = True
                    break
        if not found:
            print("  [OK] no obvious loop pattern on rank 0.")

    # C/D) Handoff consistency.
    for tag, label in (
        ("pcp_full.prev_sampled_for_scatter", "C"),
        ("pcp_full.draft_for_scatter", "D"),
    ):
        print(f"\n=== {label}) Handoff consistency ({tag}) ===")
        bad: list[int] = []
        for i, it in enumerate(iters):
            by_rank = it.get(tag, {})
            if by_rank and len(set(by_rank.values())) > 1:
                bad.append(i)
        if not iters:
            continue
        print(f"  divergent iters: {len(bad)} / {len(iters)}")
        if bad[:5]:
            print(f"  first divergent iter indices: {bad[:5]}")

    # E) When do the eagle.* fingerprints start to drift?
    print("\n=== E) eagle.* drift detection (rank 0) ===")
    print("For each eagle.* tag we look at the iter index at which the")
    print("rank-0 fingerprint FIRST differs from the dominant pattern of")
    print("the first 1/3 of iters (treated as a 'warmup baseline').")
    eagle_tags = sorted({t for it in iters for t in it if t.startswith("eagle.")})
    if not eagle_tags:
        print("  no eagle.* tags found; rerun with the extended instrumentation.")
    else:
        baseline_end = max(2, len(iters) // 3)
        for tag in eagle_tags:
            seq = [it[tag].get(0) for it in iters if tag in it]
            if len(seq) < 4:
                continue
            baseline = Counter(seq[:baseline_end])
            top = baseline.most_common(1)[0][0] if baseline else None
            if top is None:
                continue
            first_drift = next(
                (i for i, v in enumerate(seq) if v is not None and v != top),
                None,
            )
            distinct = len(set(seq))
            tag_short = tag if len(tag) <= 44 else tag[:41] + "..."
            print(
                f"  {tag_short:44s} distinct={distinct:4d}  first_drift_iter={first_drift}"
            )


def report_compare(
    iters_a: list[dict[str, dict[int, str]]],
    iters_b: list[dict[str, dict[int, str]]],
    label_a: str,
    label_b: str,
) -> None:
    print(f"\n=== Compare {label_a} vs {label_b} (rank 0) ===")
    print("For each tag present in both, find the first iter where the")
    print("rank-0 fingerprint differs between the two runs.")
    n = min(len(iters_a), len(iters_b))
    if n == 0:
        print("  no overlapping iters.")
        return
    all_tags = sorted(
        {t for it in iters_a[:n] for t in it} & {t for it in iters_b[:n] for t in it}
    )
    rows: list[tuple[str, int, int, str | None]] = []
    for tag in all_tags:
        sa = [iters_a[i].get(tag, {}).get(0) for i in range(n)]
        sb = [iters_b[i].get(tag, {}).get(0) for i in range(n)]
        diffs = [
            i for i in range(n)
            if sa[i] is not None and sb[i] is not None and sa[i] != sb[i]
        ]
        if diffs:
            rows.append((tag, len(diffs), diffs[0], None))
        else:
            rows.append((tag, 0, -1, "match"))
    # Sort: tags that diverge earliest first.
    rows.sort(key=lambda r: (r[2] if r[2] >= 0 else 10**9, r[0]))
    print(f"  {'TAG':50s} {'DIVERGE':>8s}  {'FIRST_DIFF_ITER':>15s}")
    for tag, n_diff, first, note in rows:
        tag_short = tag if len(tag) <= 50 else tag[:47] + "..."
        first_str = "-" if first < 0 else str(first)
        suffix = "" if note is None else f"  ({note})"
        print(f"  {tag_short:50s} {n_diff:>8d}  {first_str:>15s}{suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="path to the [LOOP-TRACE] log")
    ap.add_argument("--tp-size", type=int, required=True)
    ap.add_argument(
        "--compare",
        default=None,
        help="path to a second log; report first iter where its fingerprints diverge from the first log",
    )
    ap.add_argument("--label-a", default="A", help="label for the first log in compare mode")
    ap.add_argument("--label-b", default="B", help="label for the second log in compare mode")
    args = ap.parse_args()

    recs_a = parse(args.log)
    if not recs_a:
        print(f"no [LOOP-TRACE] lines found in {args.log}", file=sys.stderr)
        return 1
    iters_a = bucket_by_iter(recs_a, args.tp_size)
    print(f"# Log A: {args.log}")
    print(f"# Parsed {len(recs_a)} trace lines from log A.")
    report_single(iters_a)

    if args.compare:
        recs_b = parse(args.compare)
        if not recs_b:
            print(f"\nno [LOOP-TRACE] lines found in {args.compare}", file=sys.stderr)
            return 1
        iters_b = bucket_by_iter(recs_b, args.tp_size)
        print(f"\n# Log B: {args.compare}")
        print(f"# Parsed {len(recs_b)} trace lines from log B.")
        report_single(iters_b)
        report_compare(iters_a, iters_b, args.label_a, args.label_b)

    return 0


if __name__ == "__main__":
    sys.exit(main())
