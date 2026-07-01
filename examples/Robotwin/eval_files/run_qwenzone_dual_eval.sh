#!/bin/bash
# Launch two QwenZone evaluations in parallel: rnn mode on GPU 0, full mode on GPU 1.
#
# Usage:
#   bash examples/Robotwin/eval_files/run_qwenzone_dual_eval.sh \
#     --task click_bell --test_num 10 --mode demo_randomized [--vlm_stride 5]

cd /mnt/workspace/yama/starVLA

usage() {
    cat >&2 <<'EOF'
Usage: bash examples/Robotwin/eval_files/run_qwenzone_dual_eval.sh [options]

Options:
  --task <name>        RoboTwin task (default: beat_block_hammer)
  --test_num <n>       Episodes per task (default: 1)
  --mode <mode>        demo_clean | demo_randomized (default: demo_clean)
  --ckpt <path>        Checkpoint (default: the v2phase2_t8 steps_50000 ckpt)
  --vlm_stride <k>     LLM refresh cadence passed to BOTH runs (0=sync, >0=async) (default: 0)
  -h, --help           Show this help
EOF
}

TASK="${TASK:-beat_block_hammer}"
TEST_NUM="${TEST_NUM:-1}"
MODE="${MODE:-demo_clean}"
CKPT="${CKPT:-/mnt/workspace/yama/starVLA/results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt}"
VLM_STRIDE="${VLM_STRIDE:-0}"

while (( $# > 0 )); do
    case "$1" in
        --task)        TASK="$2"; shift 2 ;;
        --test_num)    TEST_NUM="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --ckpt)        CKPT="$2"; shift 2 ;;
        --vlm_stride)  VLM_STRIDE="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

echo "=== Launching Dual QwenZone Evaluation ==="
echo "Task: ${TASK}  TestNum: ${TEST_NUM}  Mode: ${MODE}  vlm_stride: ${VLM_STRIDE}"
echo "Checkpoint: ${CKPT}"
echo "=========================================="

# Kill any existing tmux sessions
tmux kill-session -t eval_rnn 2>/dev/null || true
tmux kill-session -t eval_full 2>/dev/null || true

# RNN mode on GPU 0 (port 5694)
echo "Starting RNN mode evaluation on GPU 0..."
tmux new-session -d -s eval_rnn -n eval_rnn \
    "cd /mnt/workspace/yama/starVLA && \
     CUDA_VISIBLE_DEVICES=0 ROBOTWIN_BASE_PORT=5694 \
     bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
       --gpu_id 0 --infer_mode rnn --vlm_stride ${VLM_STRIDE} \
       --ckpt '${CKPT}' --task ${TASK} --test_num ${TEST_NUM} --mode ${MODE}"

# Wait a bit to ensure first server starts
sleep 5

# Full mode on GPU 1 (port 5695)
echo "Starting Full mode evaluation on GPU 1..."
tmux new-session -d -s eval_full -n eval_full \
    "cd /mnt/workspace/yama/starVLA && \
     CUDA_VISIBLE_DEVICES=1 ROBOTWIN_BASE_PORT=5695 \
     bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
       --gpu_id 1 --infer_mode full --vlm_stride ${VLM_STRIDE} \
       --ckpt '${CKPT}' --task ${TASK} --test_num ${TEST_NUM} --mode ${MODE}"

echo ""
echo "=== Evaluations Started ==="
echo "RNN mode : tmux attach -t eval_rnn  (GPU 0, port 5694)"
echo "Full mode: tmux attach -t eval_full (GPU 1, port 5695)"
echo ""
echo "To monitor: tmux attach -t eval_rnn ; Ctrl+B,D to detach ; tmux attach -t eval_full"
echo "To stop:    tmux kill-session -t eval_rnn ; tmux kill-session -t eval_full"
echo "==========================="
