#!/bin/bash
# QwenZone RoboTwin eval launcher (uses starVLA start_eval.sh)
#
# Usage: bash examples/Robotwin/eval_files/run_qwenzone_eval.sh
cd /mnt/workspace/yama/starVLA

export PYTHONPATH=/mnt/workspace/yama/starVLA
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/workspace/yama/oss_yama/cache/hf_cache/hub
export STARVLA_PYTHON=/mnt/workspace/yama/miniconda3/envs/starVLA/bin/python
export ROBOTWIN_PYTHON=/mnt/workspace/yama/miniconda3/envs/RoboTwin/bin/python
export PATH=/mnt/workspace/yama/miniconda3/envs/starVLA/bin:${PATH}

# Configurable via env: CKPT, TASK, TEST_NUM, MODE
CKPT="${CKPT:-/mnt/workspace/yama/starVLA/results/Checkpoints_QwenZone/qwenzone_p2/checkpoints/steps_5000_pytorch_model.pt}"
TASK="${TASK:-beat_block_hammer}"
TEST_NUM="${TEST_NUM:-1}"
MODE="${MODE:-demo_clean}"

bash examples/Robotwin/eval_files/start_eval.sh \
  -m "${MODE}" \
  -n qwenzone_eval \
  -c "${CKPT}" \
  -t "${TEST_NUM}" \
  "${TASK}"
