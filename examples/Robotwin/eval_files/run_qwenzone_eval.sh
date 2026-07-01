#!/bin/bash
# QwenZone RoboTwin eval launcher (uses starVLA start_eval.sh)
#
# Usage (named args):
#   bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
#     --gpu_id 0 --infer_mode full --vlm_stride 0 \
#     --ckpt <path> --task click_bell --test_num 10 --mode demo_randomized
cd /mnt/workspace/yama/starVLA

export PYTHONPATH=/mnt/workspace/yama/starVLA
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/workspace/yama/oss_yama/cache/hf_cache/hub
export STARVLA_PYTHON=/mnt/workspace/yama/miniconda3/envs/starVLA/bin/python
export ROBOTWIN_PYTHON=/mnt/workspace/yama/miniconda3/envs/RoboTwin/bin/python
export PATH=/mnt/workspace/yama/miniconda3/envs/starVLA/bin:${PATH}

usage() {
    cat >&2 <<'EOF'
Usage: bash examples/Robotwin/eval_files/run_qwenzone_eval.sh [options] [<task>]

Required:
  --ckpt <path>            Checkpoint path (*.pt)

Options:
  --gpu_id <id>            GPU id (default: 0)
  --infer_mode <mode>      full (complete history) | rnn (single-step + memory)  (default: full)
  --vlm_stride <k>         0 = run the LLM every step (sync); >0 = refresh the LLM every k steps
                           and reuse cached intent in between (async, needs k-1 <= d_max)  (default: 0)
  --task <name>            RoboTwin task name (default: beat_block_hammer). May also be passed as a
                           trailing positional argument.
  --test_num <n>           Episodes per task (default: 1)
  --mode <mode>            demo_clean | demo_randomized  (default: demo_clean)
  -h, --help               Show this help

Environment:
  CUDA_VISIBLE_DEVICES     Limits GPU visibility (set this, not just --gpu_id, on multi-GPU hosts)
  ROBOTWIN_BASE_PORT       Base port for the policy server (default: 5694)

Example:
  bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
    --gpu_id 0 --infer_mode rnn --vlm_stride 5 \
    --ckpt results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt \
    --task click_bell --test_num 10 --mode demo_randomized
EOF
}

# Defaults (env overrides still honored via ${VAR:-default})
GPU_ID="${GPU_ID:-0}"
INFER_MODE="${INFER_MODE:-full}"
VLM_STRIDE="${VLM_STRIDE:-0}"
CKPT="${CKPT:-}"
TASK="${TASK:-beat_block_hammer}"
TEST_NUM="${TEST_NUM:-1}"
MODE="${MODE:-demo_clean}"

while (( $# > 0 )); do
    case "$1" in
        --gpu_id)       GPU_ID="$2"; shift 2 ;;
        --infer_mode)   INFER_MODE="$2"; shift 2 ;;
        --vlm_stride)   VLM_STRIDE="$2"; shift 2 ;;
        --ckpt)         CKPT="$2"; shift 2 ;;
        --task)         TASK="$2"; shift 2 ;;
        --test_num)     TEST_NUM="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; if (( $# > 0 )); then TASK="$1"; fi; break ;;
        -*)             echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)              TASK="$1"; shift ;;  # trailing positional = task name
    esac
done

if [[ -z "${CKPT}" ]]; then
    echo "[ERROR] --ckpt is required." >&2
    usage
    exit 1
fi

if [[ "${INFER_MODE}" != "full" && "${INFER_MODE}" != "rnn" ]]; then
    echo "[ERROR] --infer_mode must be 'full' or 'rnn' (got '${INFER_MODE}')." >&2
    exit 1
fi

# Export for policy server (read downstream by start_eval.sh / run_policy_server.sh)
export ROBOTWIN_SERVER_GPU="${GPU_ID}"
export ROBOTWIN_SERVER_PORT="${ROBOTWIN_SERVER_PORT:-5694}"
export ROBOTWIN_INFER_MODE="${INFER_MODE}"
export ROBOTWIN_VLM_STRIDE="${VLM_STRIDE}"

echo "=== QwenZone Evaluation ==="
echo "GPU_ID: ${GPU_ID}"
echo "INFER_MODE: ${INFER_MODE}"
echo "VLM_STRIDE: ${VLM_STRIDE}"
echo "CKPT: ${CKPT}"
echo "TASK: ${TASK}"
echo "TEST_NUM: ${TEST_NUM}"
echo "MODE: ${MODE}"
echo "========================"

bash examples/Robotwin/eval_files/start_eval.sh \
  -m "${MODE}" \
  -n qwenzone_eval \
  -c "${CKPT}" \
  -t "${TEST_NUM}" \
  "${TASK}"
