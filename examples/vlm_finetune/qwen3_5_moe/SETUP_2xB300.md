# Setup — Qwen3.6-35B packed SFT on a bare 2 × B300 pod

How to take a stock RunPod PyTorch image (no Docker, no CUDA toolkit, no TE) to a working
`neat` packing + TransformerEngine training run. Written from the 2026-09-21 bring-up;
every error below was hit and every fix is the one that worked. ~25 min wall-clock once the
model download is in flight.

Everything is scripted under `b300/`. Read this once, then run the scripts.

| file | what it does |
|---|---|
| `b300/provision.sh <stage…>\|all` | idempotent stages, each leaves `/workspace/.stage.<name>` |
| `b300/env.sh` | runtime env — **source before every python/automodel call** |
| `b300/buildenv.sh` | build env for the source packages (TE-torch, causal-conv1d, DeepEP) |
| `b300/rungs.sh p0\|p1\|p2\|p3` | `PACKING_BRINGUP.md` rungs P0–P3 |
| `b300/train.sh p4\|p5\|mock50\|overfit\|constlr` | smoke and mock runs |
| `b300/run_lbs.sh <lbs> <gbs> <steps> <tag> [ac]` | parameterised run; used for the memory probes and the 4/32 run |

## 0. Box

RunPod pod, 2 × B300 SXM6 (SM 10.3, 275 GB), 344 CPUs, 4 TB RAM, Ubuntu 24.04,
`torch 2.8.0+cu128` preinstalled (ignored — the venv brings its own), **no docker**, **no
nvcc**, `/workspace` persistent volume (make it ≥ 200 GB: model 72 GB + venv ~30 GB +
corpus + logs; the default 50 GB does not fit the model).

The SSH port changes on every restart. `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`. If
the key is refused right after a restart, the entrypoint has not written `authorized_keys`
yet — wait, do not debug.

## 1. Order of operations

```bash
scp b300/*.sh root@<ip>:/workspace/ ; ssh … 'chmod +x /workspace/*.sh'
cd /workspace && ./provision.sh dirs cuda            # clone + CUDA 13 toolkit    (~2 min)
# start the 72 GB model download NOW, in parallel with the builds (~35 s on this pod's link)
uv venv --python 3.12 /workspace/dlenv && VIRTUAL_ENV=/workspace/dlenv uv pip install "huggingface_hub[hf_transfer]"
HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 nohup /workspace/dlenv/bin/hf download Qwen/Qwen3.6-35B-A3B &
./provision.sh sync te ccv deepep torchao cudnn      # venv + kernels             (~12 min)
./provision.sh data short                            # prefilter v5_130k + fast-rung subset (~3 min)
./rungs.sh p1 && ./rungs.sh p2 && ./rungs.sh p3      # gates, no training yet
./train.sh p4                                        # 5-step smoke on the short subset
```

`cudnn` **must be the last install stage** (see §3.2). `data` needs the venv; it runs in
parallel with the kernel builds if started from a second shell.

What `sync` installs, and why not `--extra moe`: `uv sync --locked --extra fla --extra vlm
--extra vlm-media --group dev`. The `cuda`/`moe` extras pull mamba-ssm, nv-grouped-gemm and
tilelang (all source builds this recipe never calls) and pin TE 2.15, which is below the
fused-attention floor and would be overwritten anyway. `--group dev` is mandatory:
`FusedLinearCrossEntropy` imports `cut_cross_entropy` from there, and without it the run
dies *after* the model has loaded.

## 2. Versions that came out the other end

| | version | note |
|---|---|---|
| CUDA toolkit | 13.0.88 (apt, `cuda-nvcc-13-0` + dev libs, not the meta package) | `/usr/local/cuda-13.0`, CCCL at `include/cccl` |
| torch | 2.10.0+cu130 (from `uv.lock`) | |
| TE | **2.18.0** (`transformer-engine-cu13` wheel + `transformer-engine-torch` sdist) | lock pins 2.15 |
| cuDNN | **9.26.0.51** | lock pins 9.15.1.9 — see §3.2 |
| FLA | 0.4.2 | has `cu_seqlens` **and** `cu_seqlens_cpu` |
| causal-conv1d | 1.6.0 | source build |
| DeepEP | `4214430` (lock commit), nvshmem 3.4.5 wheel | intranode only needed here |
| torchao | 0.14.0 `--no-deps` | 0.17 breaks AdamW8bit on FSDP2 (RUNBOOK §8.3) |

Gate, in this order — do not start a run until all three print the right thing:

```
python -c "import torch,transformer_engine as te; print(te.__version__, torch.backends.cudnn.version())"  # 2.18.0 92600
python -c "import fla, causal_conv1d, deep_ep, transformer_engine"                                       # no ImportError
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 <any thd forward at head_dim 256>  ->  "Selected backend = FusedAttention"
```

## 3. Errors, in the order they appear

### 3.1 apt: `E: Conflicting values set for option Signed-By`
The image ships an unsigned `/etc/apt/sources.list.d/cuda.list` for the same URL that
`cuda-keyring` signs. **Fix:** `rm /etc/apt/sources.list.d/cuda.list` before installing the
keyring. (`provision.sh cuda`)

### 3.2 cuDNN silently reverts to 9.15 — the one that costs a day if missed
`torch 2.10.0+cu130` declares `nvidia-cudnn-cu13==9.15.1.9` as an **exact** pin. Any
resolving `uv pip install` after you installed 9.26 puts 9.15 back; the log shows
`- nvidia-cudnn-cu13==9.26.0.51 / + nvidia-cudnn-cu13==9.15.1.9`. Below 9.23 TE picks
`UnfusedDotProductAttention` at `head_dim 256` on SM 10.x, which is +259 GiB at 32k tokens
— it presents as an **OOM at pack_size 40960**, not as a version error.
**Fix:** install cuDNN **last**, with `--no-deps`, and assert:
`python -c "import torch; assert torch.backends.cudnn.version() >= 92300"`. (`provision.sh cudnn`)

### 3.3 uv: `` `torch` was declared as an extra build dependency with `match-runtime = true`, but was not found in the resolution ``
The repo's `pyproject.toml` has `[tool.uv.extra-build-dependencies] deep_ep = [{ requirement
= "torch", match-runtime = true }]`; with `--no-deps` torch is absent from the resolution.
**Fix:** run the DeepEP `uv pip install` from **outside the repo** (`cd /workspace`) with
`VIRTUAL_ENV=/workspace/Automodel/.venv`. RUNBOOK §5 already said "from outside the repo";
this is why.

### 3.4 DeepEP: `fatal error: infiniband/mlx5dv.h: No such file or directory`
IBGDA sources need rdma-core headers even for a single node, and `DISABLE_NVSHMEM=1` is
**not** honoured by the pinned hybrid-ep commit (the retry still passes nvshmem includes).
**Fix:** `apt-get install libibverbs-dev librdmacm-dev`.

### 3.5 DeepEP: `/usr/bin/ld: cannot find -l:libnvshmem_host.so`
The `nvidia-nvshmem-cu13` wheel ships only `libnvshmem_host.so.3`; DeepEP links the exact
filename. **Fix:** `ln -s libnvshmem_host.so.3 <venv>/…/nvidia/nvshmem/lib/libnvshmem_host.so`.

### 3.6 `cuDNN Frontend warning: Multiple libcudart libraries found … Using libcudart.so.12`
The image's CUDA 12.8 tree under `/usr/local` leaks into the frontend's scan.
**Fix:** `export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13` (in `env.sh`).

### 3.7 Runtime env (RUNBOOK §5, restated because it is easy to lose)
`CUDNN_HOME=<venv>/lib/python3.12/site-packages/nvidia/cudnn`; venv `nvidia/cudnn/lib` and
`nvidia/nccl/lib` **first** on `LD_LIBRARY_PATH`; the toolkit's `lib64` **not** on it (its
cuBLASLt shadows the wheel's → `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`).
Build env: `TORCH_CUDA_ARCH_LIST="10.0;10.3"`, `NVTE_CUDA_ARCHS="100;103"`,
`NVTE_WITH_NCCL_EP=0`, `CPATH=$CUDA_HOME/include/cccl`. All in `env.sh` / `buildenv.sh`.

### 3.8 Harmless noise you will see
`Skipping import of cpp extensions due to incompatible torch version … torchao 0.14.0`
(expected, RUNBOOK §8.3); `grouped_gemm is not available` (only `experts: gmm` needs it);
`Current triton version: 3.6.0, Required triton version: 3.2.0`; the qwen3_vl
`min_frames`/`max_frames` docstring `[ERROR]` lines from transformers.

## 4. Running things without stepping on yourself

- **Only one training job fits.** An EP2 35B job peaks at 150–200 GiB per GPU; two of them
  OOM or, before that, collide on torchrun's port: `EADDRINUSE … port: 29500`. Before every
  launch: `ps -eo args | grep -cE "[a]utomodel examples|[r]ecipes/vlm/finetune.py"` must be
  `0` and `nvidia-smi` must show ~0 MiB.
- **Launch detached:** `setsid nohup … > log 2>&1 < /dev/null &`. Plain `nohup … &` inside an
  `ssh` command keeps the ssh session open until the job ends (the 120 s tool timeout then
  backgrounds the *ssh*, and the job is still fine — but it is confusing).
- **Stop by PID, never by `pkill -f` pattern.** `pkill -f automodel` matches the ssh
  command line that is running it and kills your own session (exit 255). Get the tree with
  `ps -o pid= --ppid <launcher>` and kill those.
- **Two Claude sessions on one box will clobber each other** — same log filenames, same
  port, same GPUs. Agree an owner first.
- `tps` in the log is real tokens/s over all ranks, tail padding excluded (`finetune.py`,
  `num_tokens_in_batch`). It is directly comparable with the v4 bring-up log's numbers.

## 5. Short runs: override the schedule, not just the warmup

RUNBOOK §7 says any run shorter than the warmup needs `--lr_scheduler.lr_warmup_steps 1`.
That is not enough for the packed config: it has `lr_decay_style: WSD` with
`wsd_decay_steps: 222`, and `lr_decay_steps` defaults to `max_steps`. For `max_steps 50`
the anneal window starts at step `50 − 222 = −172`, i.e. **the LR is already in the tail of
the cosine at step 1** (5e-5 at step 0, 6e-6 at step 1, 5e-7 by step 49). A 50-step run
like that shows batch noise, not training. For any smoke/mock run pass
`--lr_scheduler.lr_decay_style constant` (and set `--optimizer.lr` / `--lr_scheduler.max_lr`
explicitly). `run_lbs.sh` does this.

## 6. Memory and speed on this box (packed, `ep_size 2`, full AC, AdamW8bit, real corpus)

| lbs / gbs | torch steady | `nvidia-smi` peak | warm s/step | real tok/s | verdict |
|---|---|---|---|---|---|
| 1 / 2 | 118.6 GiB | 147.6 GB | 11.6–12.7 | ~7,000 | fine |
| 4 / 8 | 165 GiB | 198.9 GB (72%) | — | 15,300 | fine |
| 4 / 32 | 193 GiB | ~226 GB (82%) | **68** | **19,200** | **use this** |
| 4 / 8, selective AC | 249 GiB | 269.0 GB (98%) | — | 15,500 (+1%) | no headroom, no gain |
| 8 / 16 | 251 GiB | 273.7 GB (99.5%) | — | 13,000–14,500 | fits by 1.3 GB; slower; do not |

Unpacked baseline on the same box (v4 bring-up, 2026-09-11): lbs 2 → ~64 s/step,
4,800–7,000 real tok/s. Packed 4/32 is the same step time with 3.5× the tokens.
