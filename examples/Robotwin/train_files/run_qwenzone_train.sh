#!/bin/bash
# QwenZone training launcher — starVLA trainer + DeepSpeed ZeRO-2
#
# Usage:
#   bash examples/Robotwin/train_files/run_qwenzone_train.sh
#   NUM_GPUS=2 MAX_STEPS=5000 bash examples/Robotwin/train_files/run_qwenzone_train.sh

cd /mnt/workspace/yama/starVLA

export PYTHONPATH=/mnt/workspace/yama/starVLA
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/workspace/yama/oss_yama/cache/hf_cache/hub
export PATH=/mnt/workspace/yama/miniconda3/envs/starVLA/bin:$PATH

NUM_GPUS=${NUM_GPUS:-2}
MAX_STEPS=${MAX_STEPS:-5000}
OUTPUT_DIR=results/Checkpoints_QwenZone/qwenzone_click_bell_t4

mkdir -p ${OUTPUT_DIR}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/starvla_qwenzone_robotwin.yaml \
  --output_dir ${OUTPUT_DIR} \
  --trainer.max_train_steps ${MAX_STEPS} \
  2>&1 | tee ${OUTPUT_DIR}/train.log
