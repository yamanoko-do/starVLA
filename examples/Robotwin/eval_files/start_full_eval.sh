#!/bin/bash
# Start Full mode evaluation on GPU 1, port 5695

cd /mnt/workspace/yama/starVLA

export PYTHONPATH=/mnt/workspace/yama/starVLA
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/workspace/yama/oss_yama/cache/hf_cache/hub
export STARVLA_PYTHON=/mnt/workspace/yama/miniconda3/envs/starVLA/bin/python
export ROBOTWIN_PYTHON=/mnt/workspace/yama/miniconda3/envs/RoboTwin/bin/python
export PATH=/mnt/workspace/yama/miniconda3/envs/starVLA/bin:${PATH}

# Fixed config for Full mode evaluation - use only GPU 1
export CUDA_VISIBLE_DEVICES=1
export ROBOTWIN_SERVER_PORT=5695
export ROBOTWIN_INFER_MODE=full

echo "=== Full Mode Evaluation ==="
echo "GPU: 1"
echo "Port: 5695"
echo "Infer Mode: full"
echo "=========================="

bash examples/Robotwin/eval_files/start_eval.sh \
  -m demo_randomized \
  -n qwenzone_eval \
  -c /mnt/workspace/yama/starVLA/results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt \
  -t 1 \
  click_bell
