# Eagle3 + DCP multi-step draft debug guide

Branch: `debug/eagle-dcp-multistep-slot` (base: `cdfd642a`).

## Enable logging

```bash
export VLLM_ASCEND_DEBUG_EAGLE_DCP=1
# optional: cap log lines (default 64)
export VLLM_ASCEND_DEBUG_EAGLE_DCP_MAX_LOGS=128
```

Logs use `logger.warning` with prefix `[EagleDcpDebug]`.

## Key log tags

| Tag | When |
|-----|------|
| `mtp_slot_pad_built` | `generate_pcp_mtp_input` builds multi-step MTP slots |
| `propose_draft_inputs` | Target aux hidden / tokens fed to Eagle |
| `topology` | CP ranks at multi-step draft init |
| `cp_multistep_init` | `slot_idx_base`, `slot_indices`, `mtp_slot_pad` head |
| `cp_slot_update` | Per `draft_step>=1` slot_mapping / `cp_seq_len` |
| `draft_first_pass` / `draft_first_pass_logits` | Eagle step 0 |
| `draft_step_forward` / `draft_step_logits` | Eagle step 1+ |
| `target_forward` | Target model forward |

## Compare DCP=1 vs DCP=2

Run the same bad prompt twice; diff logs for `cp_slot_update` and `mtp_slot_pad_built` at `draft_step=1`.

On TP16 + DCP2, collect logs from **both** `dcp_rank=0` and `dcp_rank=1` (one rank per NPU process / log file).

## Minimal repro matrix

1. DCP=2, `num_speculative_tokens=3` — expect loop
2. DCP=2, `num_speculative_tokens=1` — expect OK
3. DCP=1, `num_speculative_tokens=3` — golden for slot values

Use greedy (`temperature=0`) on a single failing prompt.
