#!/usr/bin/env bash
# Shared local/Slurm launcher. Configure methods and paths through environment variables.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -n "${VENV_PATH:-}" ]]; then
    source "${VENV_PATH}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-1}"
WHICH="${WHICH:-eulerian}"
STOPGRAD_MODE="${STOPGRAD_MODE:-simp}"
SEED="${SEED:-0}"
COMPILE="${COMPILE:-1}"
VELOCITY_WARMUP_STEPS="${VELOCITY_WARMUP_STEPS:-40000}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
BATCH_SIZE="${BATCH_SIZE:-128}"
USE_WANDB="${USE_WANDB:-0}"

case "${WHICH}" in
    eulerian|lagrangian|meanflow|flowmatching) ;;
    *) echo "Unsupported WHICH: ${WHICH}" >&2; exit 2 ;;
esac
case "${STOPGRAD_MODE}" in
    none|orig|simp) ;;
    *) echo "STOPGRAD_MODE must be none, orig, or simp." >&2; exit 2 ;;
esac
for value in "${SEED}" "${VELOCITY_WARMUP_STEPS}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Seed and warmup steps must be nonnegative integers." >&2; exit 2; }
done
for value in "${GPUS}" "${BATCH_SIZE}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS and BATCH_SIZE must be positive integers." >&2; exit 2; }
done
[[ "${COMPILE}" =~ ^[01]$ && "${USE_WANDB}" =~ ^[01]$ ]] || { echo "COMPILE and USE_WANDB must be 0 or 1." >&2; exit 2; }

# Only the ESD/LSD objectives have a separate velocity-only warmup phase.
if [[ "${WHICH}" == meanflow || "${WHICH}" == flowmatching ]]; then
    VELOCITY_WARMUP_STEPS=0
fi
if [[ "${WHICH}" =~ ^(eulerian|lagrangian)$ && "${STOPGRAD_MODE}" != simp ]]; then
    COMPILE_SECOND_ORDER="${COMPILE_SECOND_ORDER:-${COMPILE}}"
else
    COMPILE_SECOND_ORDER="${COMPILE_SECOND_ORDER:-0}"
fi
[[ "${COMPILE_SECOND_ORDER}" =~ ^[01]$ ]] || { echo "COMPILE_SECOND_ORDER must be 0 or 1." >&2; exit 2; }
if [[ "${COMPILE_SECOND_ORDER}" == 1 && "${COMPILE}" != 1 ]]; then
    echo "COMPILE_SECOND_ORDER=1 requires COMPILE=1." >&2
    exit 2
fi

JOBID="${SLURM_JOB_ID:-0}"
RUN_NAME="${RUN_NAME:-${WHICH}_${STOPGRAD_MODE}_warmup${VELOCITY_WARMUP_STEPS}_seed${SEED}}"
CKPT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/${JOBID}/${SEED}/ckpts"
PREEMPT_SENTINEL="${CKPT_DIR}/preempt-checkpoint-complete"
COMPLETION_SENTINEL="${CKPT_DIR}/training-complete"
mkdir -p "${CKPT_DIR}"
if [[ -f "${COMPLETION_SENTINEL}" ]]; then
    echo "Training already completed: ${CKPT_DIR}. Use a new RUN_NAME for a fresh run."
    exit 0
fi
rm -f "${PREEMPT_SENTINEL}"
export STOPGRAD_PREEMPT_SENTINEL="${PREEMPT_SENTINEL}"
export STOPGRAD_COMPLETION_SENTINEL="${COMPLETION_SENTINEL}"

RESUME_ARGS=()
CKPT_PATH="$("${PYTHON_BIN}" scripts/select_resume_checkpoint.py "${CKPT_DIR}")"
if [[ -z "${CKPT_PATH}" && -n "${RESUME_CHECKPOINT:-}" ]]; then
    [[ -s "${RESUME_CHECKPOINT}" ]] || { echo "Missing RESUME_CHECKPOINT: ${RESUME_CHECKPOINT}" >&2; exit 2; }
    CKPT_PATH="${RESUME_CHECKPOINT}"
fi
if [[ -n "${CKPT_PATH}" ]]; then
    echo "Resuming from ${CKPT_PATH}"
    RESUME_ARGS+=(--resume "${CKPT_PATH}")
elif [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then
    echo "No checkpoint found for requeued job; refusing to restart from step zero." >&2
    exit 3
fi

if [[ "${USE_WANDB}" == 1 ]]; then
    : "${WANDB_ENTITY:?Set WANDB_ENTITY when USE_WANDB=1}"
    export WANDB_PROJECT="${WANDB_PROJECT:-stopgrad}"
    export WANDB_RUN_ID="${WANDB_RUN_ID:-${JOBID}-${RUN_NAME}}"
    export WANDB_RESUME=allow
fi

LAUNCHER=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCHER=(srun --ntasks=1 --ntasks-per-node=1)
fi
set +e
"${LAUNCHER[@]}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${GPUS}" \
    main.py \
    --run_name "${RUN_NAME}" --job_id "${JOBID}" --task_id "${SEED}" \
    --seed "${SEED}" --which "${WHICH}" --stopgrad_type "${STOPGRAD_MODE}" \
    --batch_size "${BATCH_SIZE}" --data_path "${DATA_ROOT}" --output_dir "${OUTPUT_ROOT}" \
    --velocity_warmup_steps "${VELOCITY_WARMUP_STEPS}" --max_train_steps 200100 \
    --compile "${COMPILE}" --compile_second_order "${COMPILE_SECOND_ORDER}" \
    --use_wandb "${USE_WANDB}" "${RESUME_ARGS[@]}" "$@"
TRAIN_STATUS=$?
set -e

if [[ "${TRAIN_STATUS}" -eq 0 ]]; then
    # The trainer writes this marker after saving its final checkpoint.
    echo "Run finished successfully."
elif [[ -f "${PREEMPT_SENTINEL}" && -n "${SLURM_JOB_ID:-}" ]]; then
    rm -f "${PREEMPT_SENTINEL}"
    echo "Emergency checkpoint saved; requeueing job ${JOBID}."
    scontrol requeue "${JOBID}"
    exit 0
fi
exit "${TRAIN_STATUS}"
