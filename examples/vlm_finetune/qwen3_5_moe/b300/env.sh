# Runtime environment for the Qwen3.6-35B packed-SFT bring-up on 2 x B300 (SM 10.x).
# Source this before any python / automodel invocation.  RUNBOOK.md section 5.
export REPO=/workspace/Automodel
export VENV=$REPO/.venv
export SITE=$VENV/lib/python3.12/site-packages

export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# TE prefers a "system" cuDNN; point it at the venv wheel or it silently selects
# UnfusedDotProductAttention (RUNBOOK section 6).
export CUDNN_HOME=$SITE/nvidia/cudnn
# The base image carries CUDA 12.8 under /usr/local, so the cuDNN frontend finds both
# libcudart.so.12 and .so.13 and warns that it picked the 12. Pin the CUDA 13 runtime
# that torch and TE are actually built against.
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13
# The venv's cudnn/nccl must come FIRST, and the CUDA toolkit's lib64 must NOT be here:
# its cuBLASLt shadows the wheel's and TE fails to import.
export LD_LIBRARY_PATH=$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/cublas/lib:${LD_LIBRARY_PATH_BASE:-}

export PATH=$VENV/bin:$PATH
cd $REPO
