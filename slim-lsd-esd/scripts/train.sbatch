#!/usr/bin/env bash
# Override resources on submission: sbatch --account=... --partition=... --gres=gpu:4 ...
#SBATCH --job-name=stopgrad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=TERM@300
set -euo pipefail

# Slurm executes a spool copy, so locate the training directory from the submission directory.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from slim-lsd-esd or set REPO_ROOT}}"
export GPUS="${GPUS:-4}"
exec bash "${REPO_ROOT}/scripts/train.sh" "$@"
