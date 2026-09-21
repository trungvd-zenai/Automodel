#!/usr/bin/env bash
# Usage: run_lbs.sh <lbs> <gbs> <max_steps> <tag> [ac_mode: true|selective]
# Packed Qwen3.6-35B on 2 x B300, LR 5e-5 held constant after a 1-step warmup (the
# config's WSD with wsd_decay_steps 222 would anneal immediately on a 30-step run).
set -u
LBS=$1; GBS=$2; STEPS=$3; TAG=$4; AC=${5:-true}
source /workspace/env.sh
export WANDB_MODE=disabled
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > /workspace/logs/memlog_${TAG}.csv &
SMI=$!
automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml \
  --nproc-per-node 2 --distributed.ep_size 2 \
  --dataset.path_or_dataset data/v5_130k_filtered/train.parquet \
  --validation_dataset.path_or_dataset data/v5_130k_filtered/val.parquet \
  --step_scheduler.local_batch_size ${LBS} --step_scheduler.global_batch_size ${GBS} \
  --step_scheduler.max_steps ${STEPS} --step_scheduler.val_every_steps 1000 \
  --optimizer.lr 5.0e-5 --lr_scheduler.max_lr 5.0e-5 \
  --lr_scheduler.lr_warmup_steps 1 --lr_scheduler.lr_decay_style constant \
  --distributed.activation_checkpointing ${AC} \
  --checkpoint.enabled false
echo "EXIT CODE $?"
kill $SMI 2>/dev/null
