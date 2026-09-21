#!/usr/bin/env bash
# PACKING_BRINGUP.md rungs P1-P3 on the 2 x B300 box.  Usage: rungs.sh <p0|p1|p2|p3>
set -uo pipefail
source /workspace/env.sh
LOGS=/workspace/logs
mkdir -p "$LOGS"

p0() {
  python -m pytest \
    tests/unit_tests/models/qwen3_5_moe/test_qwen3_5_moe_packed_te_attention.py \
    tests/unit_tests/datasets/vlm/test_packed_te_labels_and_mask.py \
    tests/unit_tests/loss/test_mtp_packed_seq_idx.py \
    tests/unit_tests/models/qwen3_5_moe/test_qwen3_5_moe_block_forward.py -q
}

p1() {
  echo "### 0. the hard gate ###"
  NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 python -c "
import transformer_engine, torch
print('TE  ', transformer_engine.__version__)
print('cuDNN', torch.backends.cudnn.version())
print('torch', torch.__version__, torch.version.cuda)
print('sm   ', torch.cuda.get_device_capability(0))
"
  echo "### kernels present ###"
  python -c "import fla, causal_conv1d, deep_ep, transformer_engine; print('all four import')"
  echo "### FLA packed signature ###"
  python -c "
import inspect
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f
sig = inspect.signature(f)
print(sig)
assert 'cu_seqlens' in sig.parameters, 'FLA build predates packed GDN support'
print('cu_seqlens_cpu present:', 'cu_seqlens_cpu' in sig.parameters)
"
  echo "### 6-D mask diagnostic (what TE did with the OLD malformed mask) ###"
  python - <<'PY'
import torch
from transformer_engine.pytorch import DotProductAttention
dpa = DotProductAttention(num_attention_heads=4, kv_channels=256, attn_mask_type="padding_causal")
q = torch.randn(1, 16, 4, 256, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, 1, 1, 1, 16, 16, dtype=torch.bool, device="cuda")
try:
    out = dpa(q, q, q, attention_mask=mask, attn_mask_type="padding_causal")
    print("RESULT: TE ACCEPTED a 6-D mask -> the old neat+te path was bleeding silently")
except Exception as exc:
    print("RESULT: TE rejected it:", type(exc).__name__, str(exc)[:400])
PY
}

p2() {
  echo "### masking, both paths (v5_130k has reasoning_content -> section 3 case 2) ###"
  python examples/vlm_finetune/qwen3_5_moe/affine/check_masking.py --dataset vuhaian/v5_130k --n 16
  echo "### config resolves the packing block ###"
  python -c "
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.recipes._typed_config import RecipeConfig
dl = RecipeConfig(load_yaml_config('examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml')).vlm_dataloader
print('packing         ', dl.packing.packing_format, dl.packing.pack_size)
print('hook            ', dl.pretokenization.label_post_hook.__name__)
print('inject_fake_imgs', dl.pretokenization.inject_fake_images)
print('sampler         ', dl.length_grouped_sampler)
"
}

p3() {
  # --perturb-neighbors is the authoritative leak test in every dtype; the packed-vs-solo
  # comparison is a gate only in fp32, where a difference cannot be rounding.
  for spec in "te float32" "sdpa float32" "te bfloat16" "sdpa bfloat16"; do
    set -- $spec
    echo "######## parity: attn=$1 dtype=$2 ########"
    python examples/vlm_finetune/qwen3_5_moe/affine/check_packed_parity.py \
        --attn "$1" --head-dim 256 --dtype "$2" --perturb-neighbors
    echo "exit=$?"
  done
}

"$1" 2>&1 | tee "$LOGS/$1.log"
