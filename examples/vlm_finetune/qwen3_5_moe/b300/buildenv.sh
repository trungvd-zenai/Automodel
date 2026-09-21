# Build-time environment for the source packages (TE-torch, causal-conv1d, DeepEP) on SM 10.x.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:/workspace/Automodel/.venv/bin:$PATH
export TORCH_CUDA_ARCH_LIST="10.0;10.3"
export NVTE_CUDA_ARCHS="100;103"
export NVTE_FRAMEWORK=pytorch
# TE's NCCL-EP sources include a header torch 2.10 lacks.
export NVTE_WITH_NCCL_EP=0
# CUDA 13 moved CCCL; DeepEP's host sources need it on the include path.
export CPATH=$CUDA_HOME/include/cccl:${CPATH:-}
export MAX_JOBS=64
export NVCC_THREADS=8
