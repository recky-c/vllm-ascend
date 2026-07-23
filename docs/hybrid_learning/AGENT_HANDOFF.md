# Hybrid Learning Debug — Agent Handoff

> **Audience:** another Cursor agent on a different Ascend server.  
> **Goal:** pull this branch → start a hybrid model → run decode → interpret `[HYBRID-*]` logs → report findings.  
> **Do not** invent new features or refactor MRv2 unless the human asks. This package is for **observability + learning**.

---

## 0. What was changed (repos)

| Repo | Branch to pull | Remote (author’s fork) | What it contains |
|------|----------------|------------------------|------------------|
| **vllm-ascend** | `feat/hybrid-learning-debug` (pull latest tip) | `https://github.com/recky-c/vllm-ascend.git` | Startup / worker / FA·GDN·MoE I/O / sample·accepted logs + **this doc** |
| **vllm** | `feat/hybrid-learning-debug` @ `1f46aca55` | `https://github.com/recky-c/vllm.git` | Scheduler trunk logs `[HYBRID-SCHED]` |

After checkout, record `git rev-parse --short HEAD` for both trees in your report.

If only **vllm-ascend** is updated on the server, worker/forward logs still work; sched logs need the **vllm** branch (or equivalent commit) installed.

**Related learning notes (same folder):**

- `hybrid_model_learning_guide.md` — conceptual walkthrough (chs 1–2.3+)
- This file — **operational** playbook for the remote agent

---

## 1. Pull instructions (run on the Ascend server)

```bash
# --- vllm-ascend ---
cd /path/to/vllm-ascend
git fetch origin   # or: git fetch fork
git checkout feat/hybrid-learning-debug
# if branch only on fork:
#   git fetch https://github.com/recky-c/vllm-ascend.git feat/hybrid-learning-debug
#   git checkout -b feat/hybrid-learning-debug FETCH_HEAD
# reinstall if needed (match site practice):
#   pip install -e .

# --- vllm (for [HYBRID-SCHED] logs) ---
cd /path/to/vllm
git fetch fork
git checkout feat/hybrid-learning-debug
# pip install -e .   # if this tree is what the server uses
```

**Verify logs are present:**

```bash
cd /path/to/vllm-ascend
rg '\[HYBRID-WORKER\]\[forward\]\[io\]' vllm_ascend/patch/worker/patch_qwen3_5.py
rg '\[HYBRID-WORKER\]\[sample\]' vllm_ascend/worker/model_runner_v1.py
rg 'reset_hybrid_fwd_log' vllm_ascend/ops/gdn.py

cd /path/to/vllm
rg '\[HYBRID-SCHED\]' vllm/v1/core/sched/scheduler.py
```

---

## 2. Environment (required for readable logs)

```bash
# Worker / forward / sample (hybrid defaults ON if unset)
export VLLM_HYBRID_WORKER_DEBUG=1
# Per-kind layer I/O budget each forward step (1 = first FA, first GDN, MoE after each, …)
export VLLM_HYBRID_FWD_LOG_N=1

# Scheduler (hybrid defaults ON if unset)
export VLLM_HYBRID_SCHED_DEBUG=1

# Optional: quieter
# export VLLM_HYBRID_FWD_LOG_N=0          # silence layer I/O only
# export VLLM_HYBRID_WORKER_DEBUG=0       # silence all worker learning logs
# export VLLM_HYBRID_SCHED_DEBUG=0
```

**Do not enable PCP / PD / exotic features** for the first decode pass unless the human asks — keep the trunk simple.

---

## 3. Start service (template — adapt model path)

Use whatever launch style the server already uses (`vllm serve`, scripts, etc.). Minimal intent:

- Model: **hybrid** Qwen3.5 / Qwen3-Next (with GDN), MoE variant OK
- Single NPU or existing TP setup is fine
- Prefer **eager** if graph capture hides logs: `--enforce-eager` when debugging

Example shape (replace paths):

```bash
export VLLM_HYBRID_WORKER_DEBUG=1
export VLLM_HYBRID_SCHED_DEBUG=1
export VLLM_HYBRID_FWD_LOG_N=1

vllm serve /path/to/Qwen3.5-MoE-or-hybrid \
  --served-model-name hybrid-debug \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --enforce-eager \
  2>&1 | tee /tmp/hybrid_learning_decode.log
```

Confirm startup lines appear:

```bash
grep '\[HYBRID-STARTUP\]' /tmp/hybrid_learning_decode.log | head
```

Expect config / groups / runner cache notes (`block_size`, multi KV groups, FA + Mamba).

---

## 4. Run decode (one short request)

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hybrid-debug",
    "prompt": "Hello",
    "max_tokens": 16,
    "temperature": 0
  }'
```

Or OpenAI chat — any API that triggers **prefill + a few decode steps**.

Keep `max_tokens` small (8–32) so the log is readable.

---

## 5. Extract & interpret logs (agent checklist)

```bash
LOG=/tmp/hybrid_learning_decode.log

echo '===== STARTUP ====='
grep '\[HYBRID-STARTUP\]' "$LOG" | head -40

echo '===== SCHED ====='
grep '\[HYBRID-SCHED\]' "$LOG" | head -40

echo '===== WORKER TRUNK ====='
grep -E '\[HYBRID-WORKER\]\[(update_states|prepare_inputs|slot_mapping|metadata|forward|logits|sample|accepted)\]' "$LOG" | head -120

echo '===== LAYER I/O ====='
grep '\[HYBRID-WORKER\]\[forward\]\[io\]' "$LOG" | head -40
```

### Expected order (one decode step)

1. `[HYBRID-SCHED][begin]` → `[alloc]` / `[end]` (multi-group `block_ids`)
2. `[HYBRID-WORKER][update_states]` — `g0`/`g1` block shapes
3. `[prepare_inputs]` → `[slot_mapping]` FA **COMPUTE** / Mamba **SKIP**
4. `[metadata]` — FA meta + `GDNAttentionMetadata` (`state_indices`)
5. `[forward][begin]` → `[forward][io]` FA / GDN / MoE → `[gdn_core]` → `[forward][end]`
6. `[logits]` → `[sample][begin/end]` → `[accepted][after_update]` (if `need_accepted_tokens`)
7. Next step: `[accepted][prepare_inputs_sync]` feeds GDN

### Pass / fail for the learning goal

| Check | Pass condition |
|-------|----------------|
| Multi KV groups | Startup or update_states shows ≥2 groups / `mamba=True` on one table |
| Slot mapping | Mamba group SKIP; FA COMPUTE |
| Metadata | ≥2 meta types (FA + GDN) |
| Layer I/O | `kind=full_attention` and `kind=linear_attention` both appear (same step or across layers) |
| MoE (if MoE model) | `kind=moe` with `after_attn_type=full_attention` and/or `linear_attention` |
| Sample loop | `[logits]` then `[sample]`; hybrid+spec shows `[accepted]` |

### What **not** to dig into on this pass

- PCP / DCP / PD connector internals  
- FA kernel micro-kernels beyond `[io] full_attention.detail`  
- MoE expert routing payloads beyond `kind=moe` shapes  

---

## 6. Log tag catalog (quick reference)

| Prefix | Phase |
|--------|--------|
| `[HYBRID-STARTUP]` | config, groups, shared pages, runner init |
| `[HYBRID-SCHED]` | schedule begin/alloc/prefix/align/end |
| `[HYBRID-WORKER][update_states]` | multi-group block_ids into batch |
| `[HYBRID-WORKER][prepare_inputs]` | positions, qsl, gdn_qsl, slot |
| `[HYBRID-WORKER][slot_mapping]` | MultiGroupBlockTable FA vs Mamba |
| `[HYBRID-WORKER][metadata]` | FA + GDN metadata build |
| `[HYBRID-WORKER][forward]` | context + layer I/O + gdn_core |
| `[HYBRID-WORKER][logits]` | hidden → logits |
| `[HYBRID-WORKER][sample]` | sampler in/out |
| `[HYBRID-WORKER][accepted]` | accepted token counts for next GDN step |

Env:

| Env | Default |
|-----|---------|
| `VLLM_HYBRID_WORKER_DEBUG` | unset → on for hybrid; `0` off; `1` on |
| `VLLM_HYBRID_SCHED_DEBUG` | same for scheduler |
| `VLLM_HYBRID_FWD_LOG_N` | `1` (per-kind budget each forward) |

---

## 7. Deliverable back to the human

After one successful decode, reply with:

1. Branch SHAs actually running (`git rev-parse HEAD` for vllm-ascend and vllm)
2. Model name + launch command used
3. A **short** paste of one full step’s grepped trunk (sched → accepted), or path to the tee log
4. Checklist table (pass/fail above)
5. Any mismatch vs `hybrid_model_learning_guide.md` expectations

---

## 8. Prompt to paste into the other Cursor chat

Copy-paste this as the user message on the Ascend machine:

```text
Read and follow exactly:
  <vllm-ascend>/docs/hybrid_learning/AGENT_HANDOFF.md

Tasks:
1) Pull/checkout the hybrid-learning-debug branches for vllm-ascend (and vllm if needed).
2) Export VLLM_HYBRID_WORKER_DEBUG=1 VLLM_HYBRID_SCHED_DEBUG=1 VLLM_HYBRID_FWD_LOG_N=1.
3) Start a hybrid Qwen3.5/Qwen3-Next (or MoE) server with --enforce-eager if practical; tee logs.
4) Send one short completions request (max_tokens<=16).
5) Grep and interpret [HYBRID-STARTUP]/[HYBRID-SCHED]/[HYBRID-WORKER] logs per the handoff checklist.
6) Report pass/fail + one step’s log excerpt. Do not enable PCP/PD unless I ask.
```

---

## 9. Code touch list (if you need to re-find hooks)

**vllm-ascend**

- `vllm_ascend/patch/platform/patch_mamba_config.py` — startup page align
- `vllm_ascend/worker/worker.py` — groups / memory
- `vllm_ascend/worker/model_runner_v1.py` — update / prepare / metadata / forward / logits / sample / accepted
- `vllm_ascend/worker/block_table.py` — slot_mapping skip
- `vllm_ascend/patch/worker/patch_qwen3_5.py` — FA / GDN / MoE I/O
- `vllm_ascend/ops/gdn.py` — `reset_hybrid_fwd_log` / `take_hybrid_fwd_log` / gdn_io / gdn_core

**vllm**

- `vllm/v1/core/sched/scheduler.py` — `[HYBRID-SCHED]`
