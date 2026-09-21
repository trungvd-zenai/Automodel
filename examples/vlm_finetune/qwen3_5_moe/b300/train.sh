#!/usr/bin/env bash
# Packed smoke / mock runs on 2 x B300.  Usage: train.sh <p4|p5|mock50>
#   p4      5 steps at pack_size 4096   -- correctness, fast iteration
#   p5      8 steps at pack_size 40960  -- production shape, memory envelope
#   mock50  50 steps at pack_size 40960 -- the run the user asked for
set -uo pipefail
source /workspace/env.sh
LOGS=/workspace/logs
mkdir -p "$LOGS"
export WANDB_MODE=disabled
CFG=examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml

# world_size 2 -> ep_size 2 keeps 256 % ep_size == 0 and dp_size * cp_size % ep_size == 0.
COMMON=(
  --nproc-per-node 2
  --distributed.ep_size 2
  --step_scheduler.local_batch_size 1
  --step_scheduler.global_batch_size 2
  --lr_scheduler.lr_warmup_steps 1
  --checkpoint.enabled false
)

case "$1" in
  p4)
    # pack_size 4096 is only valid against a corpus whose rows all fit in it.  The VLM
    # neat packer books an over-long row into a bin at a CLAMPED planning length and only
    # finds the real length in __getitem__, where it drops the sample with a bare
    # logger.warning; a bin holding only that sample then yields a padding-only pack.  The
    # LLM packer raises here, the VLM one does not.  On v5_130k (p50 3,349 / mean 4,664)
    # about half the corpus would vanish that way and num_label_tokens -- the single number
    # this rung reads -- would be meaningless.  So the SHORT subset is used, not the full
    # corpus, and drop_long_samples is set so any residual over-long row fails in stage 1
    # rather than becoming padding.
    NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 automodel "$CFG" "${COMMON[@]}" \
      --dataset.path_or_dataset data/v5_130k_short/train.parquet \
      --validation_dataset.path_or_dataset data/v5_130k_short/val.parquet \
      --packed_sequence.pack_size 4096 \
      --packed_sequence.collate_max_length 4096 \
      --packed_sequence.drop_long_samples true \
      --step_scheduler.max_steps 5 \
      --step_scheduler.val_every_steps 1000
    ;;
  p5)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > "$LOGS/memlog_p5.csv" &
    SMI=$!
    NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 automodel "$CFG" "${COMMON[@]}" \
      --dataset.path_or_dataset data/v5_130k_filtered/train.parquet \
      --validation_dataset.path_or_dataset data/v5_130k_filtered/val.parquet \
      --step_scheduler.max_steps 8 \
      --step_scheduler.val_every_steps 1000
    kill $SMI 2>/dev/null
    ;;
  mock50)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > "$LOGS/memlog_mock50.csv" &
    SMI=$!
    automodel "$CFG" "${COMMON[@]}" \
      --dataset.path_or_dataset data/v5_130k_filtered/train.parquet \
      --validation_dataset.path_or_dataset data/v5_130k_filtered/val.parquet \
      --step_scheduler.max_steps 50 \
      --step_scheduler.val_every_steps 1000
    kill $SMI 2>/dev/null
    ;;
  overfit)
    # Can this configuration learn at all?  A handful of short rows makes 2 packs, so with
    # lbs 1 on 2 GPUs and drop_last every optimizer step sees the SAME data.  At a constant
    # real LR the loss must fall towards zero; if it plateaus, masking or gradient flow is
    # broken and no amount of schedule tuning will fix it.  lr_decay_style constant is
    # essential -- the committed WSD schedule with wsd_decay_steps 222 collapses the LR
    # immediately on any run shorter than 222 steps.
    automodel "$CFG" "${COMMON[@]}" \
      --dataset.path_or_dataset data/v5_130k_tiny/train.parquet \
      --validation_dataset.path_or_dataset data/v5_130k_tiny/val.parquet \
      --packed_sequence.pack_size 4096 \
      --packed_sequence.collate_max_length 4096 \
      --packed_sequence.drop_long_samples true \
      --step_scheduler.max_steps 40 \
      --step_scheduler.val_every_steps 1000 \
      --lr_scheduler.lr_decay_style constant
    ;;
  constlr)
    # Representative shape, constant LR, long enough for a real descent to show.
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > "$LOGS/memlog_constlr.csv" &
    SMI=$!
    automodel "$CFG" "${COMMON[@]}" \
      --dataset.path_or_dataset data/v5_130k_filtered/train.parquet \
      --validation_dataset.path_or_dataset data/v5_130k_filtered/val.parquet \
      --step_scheduler.max_steps 100 \
      --step_scheduler.val_every_steps 1000 \
      --lr_scheduler.lr_warmup_steps 5 \
      --lr_scheduler.lr_decay_style constant
    kill $SMI 2>/dev/null
    ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
