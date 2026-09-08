# SPDX-License-Identifier: Apache-2.0
"""Deterministic correctness requests; use the benchmark skill for TTFT."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def request(base_url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        return response.read().decode()


def speculative_counters(metrics):
    names = ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total")
    counters = {}
    for line in metrics.splitlines():
        match = re.fullmatch(r"([^ {]+)(\{.*\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?", line)
        if match and match[1] in names:
            key = match[1] + (match[2] or "")
            if key in counters:
                raise ValueError(f"Duplicate metric sample: {key}")
            counters[key] = float(match[3])
    return counters


def build_prompts(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    vocabulary = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
    rng = random.Random(args.seed)

    def tokens(length):
        return rng.choices(vocabulary, k=length)

    cases = []
    for length in map(int, args.input_lengths.split(",")):
        if args.scenario in ("prefix", "combined", "mtp", "pp"):
            prefixes = [length // 2 // 128 * 128, int(length * 0.9) // 128 * 128, 127, 128, 129]
            for prefix_length in prefixes:
                prefix = tokens(prefix_length)
                prime = prefix + tokens(128)
                target = prefix + tokens(length - prefix_length)
                if prime[prefix_length] == target[prefix_length]:
                    target[prefix_length] = vocabulary[(vocabulary.index(target[prefix_length]) + 1) % len(vocabulary)]
                cases.append(
                    dict(
                        label=f"{length}-prefix-{prefix_length}",
                        prime=prime,
                        prompt=target,
                        expected_prefix=prefix_length // 128 * 128,
                    )
                )
        else:
            cases.append(dict(label=str(length), prompt=tokens(length)))
    if args.scenario == "mixed":
        prefix = tokens(16384)
        cases = [dict(label="mixed", prime=prefix + tokens(128), prompts=[tokens(32768), prefix + tokens(16384)])]
    return cases


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # OFF and candidates in sibling result directories reuse this exact file.
    key = f"{args.scenario}-{args.seed}-{args.input_lengths}"
    prompts_path = output.parent / f"prompts-{key}.json"
    if not prompts_path.exists():
        prompts_path.write_text(json.dumps(build_prompts(args)))
    cases = json.loads(prompts_path.read_text())
    (output / "inputs.sha256").write_text(hashlib.sha256(prompts_path.read_bytes()).hexdigest())
    metrics_before = request(args.base_url, "/metrics")
    (output / "metrics-before.txt").write_text(metrics_before)

    def complete(prompt):
        return json.loads(
            request(
                args.base_url,
                "/v1/completions",
                dict(
                    model=args.model,
                    prompt=prompt,
                    temperature=0,
                    seed=args.seed,
                    max_tokens=args.output_tokens,
                    ignore_eos=True,
                    return_token_ids=True,
                    logprobs=1,
                ),
            )
        )

    results = []
    try:
        for case in cases:
            result = {"label": case["label"]}
            if "prime" in case:
                result["prime"] = complete(case["prime"])
            if "prompts" in case:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    result["responses"] = list(executor.map(complete, case["prompts"]))
                # Actual mixed batching must be checked in worker audit output.
            else:
                response = complete(case["prompt"])
                result["response"] = response
                (output / "responses.json").write_text(json.dumps(results + [result]))
                assert response["usage"]["prompt_tokens"] == len(case["prompt"])
                assert response["usage"]["completion_tokens"] == args.output_tokens
                if case.get("expected_prefix", 0):
                    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
            results.append(result)
            (output / "responses.json").write_text(json.dumps(results))
    finally:
        metrics_after = request(args.base_url, "/metrics")
        (output / "metrics-after.txt").write_text(metrics_after)
    if args.scenario in ("mtp", "combined"):
        before = speculative_counters(metrics_before)
        after = speculative_counters(metrics_after)
        deltas = {key: value - before.get(key, 0) for key, value in after.items()}
        (output / "speculative-deltas.json").write_text(json.dumps(deltas, indent=2))
        drafted = sum(
            value for key, value in deltas.items() if key.startswith("vllm:spec_decode_num_draft_tokens_total")
        )
        accepted = sum(
            value for key, value in deltas.items() if key.startswith("vllm:spec_decode_num_accepted_tokens_total")
        )
        assert drafted > 0, "MTP did not generate draft tokens."
        assert 0 < accepted < drafted, "Acceptance and rejection both need coverage; retain this case as incomplete."
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        assert len(reference) == len(results)
        for baseline, candidate in zip(reference, results):
            for before, after in zip(
                baseline.get("responses", [baseline.get("response")]),
                candidate.get("responses", [candidate.get("response")]),
            ):
                assert before["choices"][0]["token_ids"] == after["choices"][0]["token_ids"], candidate["label"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm52-broadcast")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--scenario", choices=("smoke", "prefix", "mixed", "decode", "mtp", "pp", "combined"), required=True
    )
    parser.add_argument("--input-lengths", default="65536,131072")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", help="OFF responses.json from the same feature configuration")
    run(parser.parse_args())
