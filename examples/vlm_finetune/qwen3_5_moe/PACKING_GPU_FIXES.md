# What it took to make `neat` packing + TE train on a real GPU

The packing change (`95a422b2e`, `8500a6f3e`) was green on every CPU test and had never
touched a GPU. This records what broke when it did (2 × B300, 2026-09-21), what was fixed,
how each fix is pinned, and what the run then measured. RUNBOOK §13.5 has the design;
`SETUP_2xB300.md` has the environment; `PACKING_BRINGUP.md` is the procedure.

## 1. Two bugs in the TE routing (commit `dde1a6607`)

Both live in `_Qwen3_5MoeAttention._packed_te_attention`
(`nemo_automodel/components/models/qwen3_5_moe/model.py`). Neither is reachable on CPU:
nothing there calls TransformerEngine, and the unit tests stub the attention function with
one that happens to have the same contract as the wrong code.

### 1.1 `cu_seqlens` were int64
```
AssertionError: cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!
  transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py:1359
```
The packed metadata is built once per forward by the shared
`prepare_gated_delta_packed_metadata` and consumed by all 30 GatedDeltaNet layers, whose FLA
kernel takes a `LongTensor`. Changing the shared dtype would touch them all, so the cast is
local to the TE call:
```python
cu_seqlens = metadata.cu_seqlens.to(torch.int32)
```
Pinned by `test_cu_seqlens_are_int32_for_te`, which also asserts the shared metadata stays
int64.

### 1.2 The repad assumed a 3-D attention output
```
IndexError: tuple index out of range        # attn_output.shape[2]
```
TE returns `[tokens, heads * head_dim]` for `qkv_format="thd"` — 2-D. The stubs returned 3-D.
The caller (`qwen3_next/layers.py:380`) only needs something reshapeable to
`[batch, sequence, -1]`, so the trailing dims are now carried through instead of assumed:
```python
return padded.reshape(batch, seq_len, *attn_output.shape[1:])
```
Pinned by `test_two_dimensional_te_output_is_repadded`, whose stub returns 2-D.

## 2. The parity gate was asking the wrong question

`affine/check_packed_parity.py` compared a packed forward with each document forwarded
alone. On GPU, in bf16, that reported document 2 as a **`LEAK` at 0.125**, in both TE and
SDPA, while documents 1 and 3 were bit-identical and everything passed in fp32 (~1e-05).

That pattern is not a leak. Packed-vs-solo conflates two things: information crossing a
document boundary, and **arithmetic reordering** — grouped-GEMM groups tokens per expert
differently in a 122-token pack than in a 61-token solo forward, and GDN chunk offsets move.
Reordering changes rounding, not results; in bf16 one ulp at these activation magnitudes is
already ~1e-2, so the 2e-2 `atol` was structurally unpassable.

The test that isolates leakage holds the layout fixed and varies only **content**: re-run the
same pack with every *other* document's hidden states re-randomised and require the document
under test to be bit-identical. That is `--perturb-neighbors`, and it is now the authoritative
result:

| run | packed vs solo | neighbour perturbation |
|---|---|---|
| `te` fp32 | 1.14e-05 / 6.62e-06 / 9.82e-06 | **0.000e+00** ×3 |
| `sdpa` fp32 | 1.16e-05 / 6.59e-06 / 9.82e-06 | **0.000e+00** ×3 |
| `te` bf16 | 0 / **0.125** (4/61 tokens) / 0 | **0.000e+00** ×3 |
| `sdpa` bf16 | 0 / **0.125** (4/61 tokens) / 0 | **0.000e+00** ×3 |

The script now separates `DRIFT` (packed-vs-solo beyond `atol`; a gate only in fp32, where
it cannot be rounding) from `LEAK` (perturbation; a gate in every dtype), and prints how many
tokens exceed `atol` so one flipped routing decision is distinguishable from a broad leak.

Two controls that were tried and are **not** valid, so nobody repeats them:
`--fake-balanced-gate` routes experts by *position* and therefore breaks packed/solo
equivalence on purpose; and "solo with trailing padding vs solo without" came out bitwise
equal, which rules out shape sensitivity but says nothing about the packed layout.

## 3. End-to-end on the real model (`affine/check_packed_e2e.py`)

The parity script runs a 4-layer stack with random weights. This one drives the recipe's own
dataloader (prefilter → knapsack → `neat_packed_vlm_collater` → `last_turn_label_hook`) into
the real 35B checkpoint and compares per-document CE against the same document forwarded
alone with **no** `position_ids`, so the model derives positions exactly as unpacked training
does.

- `position_ids` in the pack: shape `[3, 1, 4096]`, value **0 at every document start**.
- fp32, TE: relative CE difference 0.16–1.07% per document, signed mean **−0.0009** over 8
  documents, no consistent direction.
- **Document 1 of every pack is the noise floor.** It sees exactly the solo context and still
  differs by 0.3–0.5%: a 4096-token forward tiles GEMMs differently from a 999-token one, and
  MoE routing amplifies it. Documents 2–4 sit in the same band. Packing adds nothing
  measurable above that floor.

## 4. Two things that were wrong in the *procedure*, not the code

### 4.1 `pack_size 4096` against the full corpus silently trains on padding
The VLM packer books an over-long row at a **clamped** planning length
(`neat_packing_vlm.py:817`), then drops it in `__getitem__` behind a bare `logger.warning`
(`:584`); a bin that held only that row returns a padding-only pack (`:616`). The LLM packer
raises here; the VLM one does not. On v5_130k (p50 3,349 tokens) about half the corpus would
vanish and `num_label_tokens` — the one number the smoke rung reads — would be meaningless.
**Fix:** shrink the corpus, not the pack. `affine/make_short_subset.py` keeps rows ≤ 1,000
tokens so nothing can be dropped and ~4 documents share each 4096 pack. 40,960 is the floor
for the real corpus (RUNBOOK §2). `dropped 0` in the packer log confirms it.

### 4.2 The 50-step "mock" run had no learning rate
`--lr_scheduler.lr_warmup_steps 1` with the config's `wsd_decay_steps: 222` on a 50-step run
starts the anneal at step −172: LR 5e-5 at step 0, 6e-6 at step 1, 5e-7 by step 49. The
curve it produced (sd 0.15 around 0.82, `corr(loss, grad_norm) = +0.63`) is batch difficulty
on a frozen model, not instability — but it also proves nothing about training. **Fix:**
short runs pass `--lr_scheduler.lr_decay_style constant`. `b300/run_lbs.sh` does.

## 5. What the corrected run measured

Packed, `lbs 4 / gbs 32`, LR 5e-5 constant after a 1-step warmup, full AC, v5_130k, 2 × B300:

| | |
|---|---|
| packs | 128,575 rows → 14,644 at 100.0% estimated utilization |
| TE | `Selected backend = FusedAttention (sub-backend 1)`, `thd_thd_thd`, `head_dim 256` |
| loss | 0.818 (step 0) → 0.54–0.56 by step 15–20, grad norm 5.96 → ~1.0 |
| supervised tokens/step | 65k–96k (mean 78k) — last-turn masking intact under packing |
| memory | 193 GiB torch steady, ~226 GB `nvidia-smi` |
| speed | **68 s/step, 19,200 real tok/s**, 1.31 M real tokens/step |

Against the unpacked baseline on the same box (v4 bring-up, lbs 2: ~64 s/step, 4,800–7,000
real tok/s): same step time, 3.5× the tokens — **2.7–3.8× faster**, and a lower bound, since
the baseline was measured on the longer-row corpus, which flatters the unpacked path. An
epoch of v4 goes from ~48 h to ~15 h on two GPUs.

Also answered: TE **raises** on the 4-D and 6-D masks the old code produced (a 2-D padding
mask returns normally), so the upstream `neat` recipes switched to `attn: te` crashed rather
than bleeding. The §2 caveat that a 2-D *indexed* map is silently accepted still stands.

## 6. Still open before the 4-node run

- **The committed step schedule is for the wrong corpus.** 14,644 packs is v5_130k; the
  config names v6_137k (≈13,770 packs). Pick the corpus, read the packer's line from a
  1-step run, then apply PACKING_BRINGUP P6.
- **Batch size and LR are hyperparameter changes hiding inside the packing change.** The
  packed config's `lbs 2 / gbs 64` is ~4.4× the documents per step of the unpacked recipe,
  at the same 5e-5. The 32-GPU floor is `lbs 1 / gbs 32` (~2.2×). Decide this explicitly.
- Rung 7 (200 steps, kill-and-resume) and the 4-node run itself have not been done packed.
