#!/usr/bin/env bash
# Provision the 2 x B300 box for the Qwen3.6-35B `neat` packing + TE bring-up.
# Stages are idempotent: each writes /workspace/.stage.<name> on success and is skipped
# on re-run.  Usage: provision.sh <stage> [<stage> ...]   or   provision.sh all
set -euo pipefail

REPO=/workspace/Automodel
BRANCH=trungvd-zenai/feat/qwen3-6-neat-packing-te
FORK=https://github.com/trungvd-zenai/Automodel.git
LOGS=/workspace/logs
mkdir -p "$LOGS" /workspace/hf /workspace/aptcache

done_marker() { echo "/workspace/.stage.$1"; }
have()        { [ -f "$(done_marker "$1")" ]; }
mark()        { touch "$(done_marker "$1")"; echo "=== STAGE $1 OK ==="; }

stage_dirs() {
  have dirs && { echo "skip dirs"; return; }
  if [ ! -d "$REPO/.git" ]; then
    git clone --branch "$BRANCH" --single-branch "$FORK" "$REPO"
  fi
  cd "$REPO" && git log --oneline -2
  mark dirs
}

stage_cuda() {
  have cuda && { echo "skip cuda"; return; }
  # The base image ships an unsigned cuda.list for the same URL the keyring signs,
  # which apt rejects as "Conflicting values set for option Signed-By".
  rm -f /etc/apt/sources.list.d/cuda.list
  local kr=/workspace/cuda-keyring.deb
  curl -fsSL -o "$kr" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i "$kr"
  apt-get -o Dir::Cache::archives=/workspace/aptcache update
  DEBIAN_FRONTEND=noninteractive apt-get -o Dir::Cache::archives=/workspace/aptcache \
    install -y --no-install-recommends \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 cuda-crt-13-0 \
      cuda-nvrtc-dev-13-0 cuda-nvml-dev-13-0 cuda-profiler-api-13-0 cuda-nvtx-13-0 \
      cuda-driver-dev-13-0 libcublas-dev-13-0 libcurand-dev-13-0 \
      libcusparse-dev-13-0 libcusolver-dev-13-0 \
      build-essential ninja-build \
      libibverbs-dev librdmacm-dev
  rm -rf /workspace/aptcache/*.deb
  /usr/local/cuda-13.0/bin/nvcc --version | tail -2
  ls -d /usr/local/cuda-13.0/include/cccl
  mark cuda
}

stage_sync() {
  have sync && { echo "skip sync"; return; }
  cd "$REPO"
  # Pass 1: torch + the pure-wheel surface this recipe uses.  The `cuda`/`moe` extras are
  # NOT installed wholesale -- they carry source builds this recipe never calls
  # (mamba-ssm, nv-grouped-gemm, tilelang) and a TE pin (2.15) that is too old for
  # head_dim 256 fused attention on SM 10.x.  The pieces that ARE needed are installed at
  # their own pins in the `te`, `ccv` and `deepep` stages.
  # --group dev is mandatory: FusedLinearCrossEntropy imports cut_cross_entropy, which
  # lives only there, and the run would die AFTER the model loads (RUNBOOK section 5).
  uv sync --locked --extra fla --extra vlm --extra vlm-media --group dev
  .venv/bin/python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
  uv pip install ninja "huggingface_hub[hf_transfer]"
  mark sync
}

stage_te() {
  have te && { echo "skip te"; return; }
  cd "$REPO"
  source /workspace/buildenv.sh
  # RUNBOOK section 6: head_dim 256 fused attention on SM 10.x needs TE >= 2.18 AND
  # cuDNN >= 9.23.  The lockfile pins TE 2.15 / cuDNN 9.15, which fall back to
  # UnfusedDotProductAttention (+259 GiB at 32k tokens).
  uv pip install nvidia-ml-py "onnxscript>=0.5.6" "nvidia-cudnn-frontend>=1.27.0"
  uv pip install --no-build-isolation \
      "transformer-engine==2.18.0" "transformer-engine-cu13==2.18.0" "transformer-engine-torch==2.18.0"
  mark te
}

stage_ccv() {
  have ccv && { echo "skip ccv"; return; }
  cd "$REPO"
  source /workspace/buildenv.sh
  uv pip install --no-build-isolation "causal-conv1d==1.6.0"
  mark ccv
}

stage_deepep() {
  have deepep && { echo "skip deepep"; return; }
  # From OUTSIDE the repo, on purpose (RUNBOOK section 5).  The repo pyproject declares
  #   [tool.uv.extra-build-dependencies]  deep_ep = [{ requirement = "torch", match-runtime = true }]
  # and with --no-deps torch is absent from the resolution, so uv aborts with
  # "`torch` was declared as an extra build dependency with `match-runtime = true`, but was
  # not found in the resolution".  Outside the repo that table does not apply and the
  # already-installed torch in VIRTUAL_ENV is used directly.
  cd /workspace
  export VIRTUAL_ENV="$REPO/.venv"
  source /workspace/buildenv.sh
  uv pip install "nvidia-nvshmem-cu13==3.4.5"
  export NVSHMEM_DIR="$REPO/.venv/lib/python3.12/site-packages/nvidia/nvshmem"
  ls "$NVSHMEM_DIR"
  # The pip wheel ships only the versioned soname (libnvshmem_host.so.3) but DeepEP links
  # with -l:libnvshmem_host.so, an exact-filename request, so ld needs the plain name.
  if [ ! -e "$NVSHMEM_DIR/lib/libnvshmem_host.so" ]; then
    ln -s libnvshmem_host.so.3 "$NVSHMEM_DIR/lib/libnvshmem_host.so"
  fi
  ls -l "$NVSHMEM_DIR/lib/libnvshmem_host.so"
  # Same commit the lockfile pins.  PR #564 (internode RC-QP layout) is NOT backported
  # here: it only matters for an ep_size that spans nodes, and this is a single node.
  if ! HYBRID_EP_MULTINODE=0 uv pip install --no-build-isolation --no-deps \
        "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@42144303752422ade37f24bca9e2dde12df70e09"; then
    echo "!!! nvshmem build failed; retrying intranode-only (DISABLE_NVSHMEM=1)"
    DISABLE_NVSHMEM=1 HYBRID_EP_MULTINODE=0 uv pip install --no-build-isolation --no-deps \
        "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@42144303752422ade37f24bca9e2dde12df70e09"
  fi
  mark deepep
}

stage_torchao() {
  have torchao && { echo "skip torchao"; return; }
  cd "$REPO"
  # RUNBOOK section 8.3: torchao 0.17 breaks AdamW8bit on FSDP2 DTensor params.
  uv pip install --no-deps "torchao==0.14.0"
  .venv/bin/python -c "import importlib.metadata as m; from torchao.optim import AdamW8bit; print('torchao', m.version('torchao'))"
  mark torchao
}

stage_cudnn() {
  have cudnn && { echo "skip cudnn"; return; }
  cd "$REPO"
  # MUST be the last install.  torch 2.10.0+cu130 declares
  # nvidia-cudnn-cu13==9.15.1.9 as an EXACT pin, so every resolving `uv pip install`
  # after this one silently downgrades cuDNN back below the 9.23 floor that head_dim 256
  # fused attention needs on SM 10.x (RUNBOOK section 6).  --no-deps skips resolution;
  # the soname stays libcudnn.so.9, which is what torch and TE dlopen.
  uv pip install --no-deps "nvidia-cudnn-cu13==9.26.0.51"
  source /workspace/env.sh
  python -c "
import torch, transformer_engine
v = torch.backends.cudnn.version()
print('cuDNN', v, '| TE', transformer_engine.__version__)
assert v >= 92300, f'cuDNN {v} is below the 9.23 floor for head_dim 256 fused attention'
print('gate OK')
"
  mark cudnn
}

stage_short() {
  have short && { echo "skip short"; return; }
  source /workspace/env.sh
  # Fast-rung corpus.  `pack_size` cannot be lowered on the full corpus: the VLM packer
  # books an over-long row at a clamped planning length and then drops it in __getitem__
  # with only a warning, yielding padding-only packs.  Shrink the data instead.
  python examples/vlm_finetune/qwen3_5_moe/affine/make_short_subset.py \
      --src data/v5_130k_filtered --out data/v5_130k_short --max-tokens 1000
  mark short
}

stage_model() {
  have model && { echo "skip model"; return; }
  source /workspace/env.sh
  hf download Qwen/Qwen3.6-35B-A3B
  du -sh "$HF_HOME"
  mark model
}

stage_data() {
  have data && { echo "skip data"; return; }
  source /workspace/env.sh
  python examples/vlm_finetune/qwen3_5_moe/affine/prefilter.py \
      --dataset vuhaian/v5_130k --max-seq-len 40960 --num-proc 32 \
      --out data/v5_130k_filtered
  ls -la data/v5_130k_filtered
  mark data
}

if [ "${1:-}" = "all" ]; then
  set -- dirs cuda sync te ccv deepep torchao cudnn model data short
fi
for s in "$@"; do
  echo "########## $s  $(date -u +%H:%M:%S) ##########"
  "stage_$s"
done
echo "########## PROVISION DONE $(date -u +%H:%M:%S) ##########"
