# SPDX-License-Identifier: Apache-2.0
"""Read official benchmark JSONs; never launch or filter benchmark requests."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(manifest, output):
    rows = []
    failures = []
    groups = defaultdict(list)
    for line in manifest.read_text().splitlines():
        entry = json.loads(line)
        result_path = Path(entry["result_file"])
        source_path = Path(entry["source_manifest"])
        if not source_path.is_file():
            raise ValueError(f"Missing source manifest: {source_path}")
        result = json.loads(result_path.read_text())
        if result["status"] != "ok" or entry.get("invalid_reason"):
            failures.append(dict(entry, reason=entry.get("invalid_reason", result["status"])))
        for case in result.get("cases", []):
            length = case["input_length"]
            for run in case["per_run"]:
                if run["warmup"]:
                    continue
                raw = run["raw_result"]
                if (
                    raw["input_lens"] != [length]
                    or raw["output_lens"] != [1]
                    or raw["completed"] != 1
                    or raw.get("failed", 0)
                ):
                    failures.append(dict(entry, reason="request count, length, or success mismatch", run=run["run"]))
                for ttft in raw["ttfts"]:
                    row = dict(
                        variant=entry["variant"],
                        round=entry["round"],
                        input_tokens=length,
                        run=run["run"],
                        ttft_seconds=ttft,
                        result_file=str(result_path),
                        source_manifest=str(source_path),
                    )
                    rows.append(row)
                    groups[(entry["variant"], entry["round"], length)].append(ttft)
    stats = []
    for (variant, round_id, length), values in groups.items():
        stats.append(
            dict(
                variant=variant,
                round=round_id,
                input_tokens=length,
                count=len(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                stddev=statistics.stdev(values) if len(values) > 1 else 0,
                minimum=min(values),
                maximum=max(values),
            )
        )
    lookup = {(s["variant"], s["round"], s["input_tokens"]): s for s in stats}
    ratios = []
    for (variant, round_id, length), stat in lookup.items():
        if variant == "layer" and ("tensor", round_id, length) in lookup:
            tensor = lookup[("tensor", round_id, length)]
            ratios.append(dict(round=round_id, input_tokens=length, layer_over_tensor=stat["mean"] / tensor["mean"]))
    off_drift = {}
    for length in (65536, 131072):
        off = sorted(
            (s for s in stats if s["variant"] == "off" and s["input_tokens"] == length), key=lambda s: s["round"]
        )
        if len(off) >= 2:
            off_drift[length] = off[-1]["mean"] / off[0]["mean"] - 1
    winner = None
    archive_regressions = {}
    for length in (65536, 131072):
        archive = [row["ttft_seconds"] for row in rows if row["variant"] == "archive" and row["input_tokens"] == length]
        if archive:
            for variant in ("tensor", "layer"):
                values = [
                    row["ttft_seconds"] for row in rows if row["variant"] == variant and row["input_tokens"] == length
                ]
                if values:
                    archive_regressions[f"{variant}-{length}"] = statistics.mean(values) / statistics.mean(archive) - 1
    both_regress = any(
        archive_regressions.get(f"tensor-{length}", 0) > 0.02 and archive_regressions.get(f"layer-{length}", 0) > 0.02
        for length in (65536, 131072)
    )
    if (
        not failures
        and not both_regress
        and len(archive_regressions) == 4
        and len(ratios) >= 4
        and len(off_drift) == 2
        and all(abs(drift) <= 0.02 for drift in off_drift.values())
        and all(s["count"] >= 5 for s in stats)
    ):
        rounds = {row["round"] for row in ratios}
        layer_rounds = sum(
            all(row["layer_over_tensor"] <= 1.01 for row in ratios if row["round"] == round_id) for round_id in rounds
        )
        if layer_rounds >= 2:
            winner = "layer"
        elif any(
            sum(row["input_tokens"] == length and row["layer_over_tensor"] > 1.01 for row in ratios) >= 2
            for length in (65536, 131072)
        ):
            winner = "tensor"
    summary = dict(
        statistics=stats,
        failures=failures,
        ratios=ratios,
        off_drift=off_drift,
        archive_regressions=archive_regressions,
        provisional_winner=winner,
        note="Winner still requires archive-regression, correctness and hardware acceptance checks.",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistics.json").write_text(json.dumps(summary, indent=2))
    if rows:
        with (output / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "| Variant | Round | Input | Samples | Mean TTFT (s) | Median (s) | Stddev (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {s['variant']} | {s['round']} | {s['input_tokens']} | {s['count']} | "
        f"{s['mean']:.6f} | {s['median']:.6f} | {s['stddev']:.6f} |"
        for s in stats
    )
    lines.append(f"\nProvisional winner: {winner or 'pending'}; failed records: {len(failures)}.")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs_manifest, args.output_dir)
