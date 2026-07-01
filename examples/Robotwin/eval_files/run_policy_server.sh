#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port] [infer_mode] [vlm_stride]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
infer_mode="${4:-${ROBOTWIN_INFER_MODE:-full}}"
vlm_stride="${5:-${ROBOTWIN_VLM_STRIDE:-0}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

# Put the starvla env's bin dir on PATH for any subprocesses spawned by the server.
starvla_bin_dir="$(dirname "${star_vla_python}")"
export PATH="${starvla_bin_dir}:${PATH}"

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"
echo "[INFO] infer_mode: ${infer_mode}"
echo "[INFO] vlm_stride: ${vlm_stride}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --infer_mode "${infer_mode}" \
    --vlm_stride "${vlm_stride}" \
    "${use_bf16_flag[@]}"
