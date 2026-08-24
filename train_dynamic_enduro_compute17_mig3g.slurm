#!/bin/bash

##### RESOURCE CONFIG: compute17 H200 3g.71gb MIG #####

#SBATCH --partition=h200mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=bml-compute17
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=63300M
#SBATCH --job-name=dyn-enduro-3g
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/work/jeongmin/thinker-dynamic/slurm_logs/slurm-%x-%j.out
#SBATCH --error=/home/work/jeongmin/thinker-dynamic/slurm_logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT=""
for candidate in \
    /home/work/jeongmin/thinker-dynamic \
    /work/jeongmin/thinker-dynamic \
    /home/jeongmin/thinker-dynamic; do
    if [ -d "$candidate/thinker" ]; then
        PROJECT_ROOT="$candidate"
        break
    fi
done
if [ -z "$PROJECT_ROOT" ]; then
    echo "[setup] thinker-dynamic project root is not visible on $(hostname)"
    exit 2
fi

PYTHON_ENV="/home/jeongmin/miniconda3/envs/thinker"
PYTHON_BIN="$PYTHON_ENV/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "[setup] missing Python environment: $PYTHON_BIN"
    exit 2
fi

JOBID="$SLURM_JOB_ID"
JOBTMP="/scratch/$USER/dt-$JOBID"
WORKDIR="$JOBTMP/p"
RUN_DIR="$WORKDIR/thinker"
XPID="dynamic-enduro-v5-test-$JOBID"
PERIODIC_STAGEOUT_SEC=300
RAY_RUNTIME_DIR="/tmp/r$JOBID"
TASK_TMPDIR="/tmp/t$JOBID"
WANDB_RUNTIME_DIR="$JOBTMP/w"

mkdir -p \
    "$WORKDIR" \
    "$RAY_RUNTIME_DIR" \
    "$TASK_TMPDIR" \
    "$WANDB_RUNTIME_DIR/cache" \
    "$WANDB_RUNTIME_DIR/data"
export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$RUN_DIR:${PYTHONPATH:-}"
export TMPDIR="$TASK_TMPDIR"
export RAY_TMPDIR="$RAY_RUNTIME_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_USER=jmme425
export WANDB_DIR="$WANDB_RUNTIME_DIR"
export WANDB_CACHE_DIR="$WANDB_RUNTIME_DIR/cache"
export WANDB_DATA_DIR="$WANDB_RUNTIME_DIR/data"
export WANDB_SILENT=true
# Authenticate through ~/.netrc so the API key is never copied into run
# metadata or exposed in the Slurm environment.
unset WANDB_API_KEY WANDB_MODE WANDB_DISABLED

echo "[meta] HOST=$(hostname)"
echo "[meta] JOBID=$JOBID"
echo "[meta] PROJECT_ROOT=$PROJECT_ROOT"
echo "[meta] WORKDIR=$WORKDIR"
echo "[meta] XPID=$XPID"
echo "[meta] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "[meta] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "[meta] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "[meta] SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
echo "[meta] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "[meta] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "[stage-in] syncing current Dynamic Thinker source"
rsync -a --delete \
    --exclude=.git/ \
    --exclude=__pycache__/ \
    --exclude='*.pyc' \
    --exclude=build/ \
    --exclude=logs/ \
    --exclude='slurm-*.out' \
    --exclude='slurm-*.err' \
    "$PROJECT_ROOT/" "$WORKDIR/"

sync_logs() {
    if [ -d "$WORKDIR/logs/thinker" ]; then
        mkdir -p "$PROJECT_ROOT/logs/thinker"
        rsync -a --update "$WORKDIR/logs/thinker/" "$PROJECT_ROOT/logs/thinker/"
    fi
}

periodic_stageout_pid=""
stop_periodic_stageout() {
    if [ -n "${periodic_stageout_pid:-}" ] && kill -0 "$periodic_stageout_pid" 2>/dev/null; then
        kill "$periodic_stageout_pid" 2>/dev/null || true
        wait "$periodic_stageout_pid" 2>/dev/null || true
    fi
}
cleanup() {
    stop_periodic_stageout
    echo "[stage-out] syncing training logs"
    sync_logs || true
}
trap cleanup EXIT

(
    set +e
    while true; do
        sleep "$PERIODIC_STAGEOUT_SEC" || exit 0
        echo "[periodic-stage-out] syncing training logs"
        sync_logs
    done
) &
periodic_stageout_pid="$!"

cd "$RUN_DIR"
echo "[env] python=$PYTHON_BIN"
"$PYTHON_BIN" --version

echo "[preflight] importing Dynamic Thinker, Enduro-v5, and W&B"
"$PYTHON_BIN" - <<'PY'
import envpool
import thinker.cenv
import wandb

env = envpool.make_gymnasium("Enduro-v5", num_envs=1, seed=1)
obs, _ = env.reset()
print("[preflight] cenv=", thinker.cenv.__file__)
print("[preflight] Enduro observation=", obs.shape, obs.dtype)
env.close()

api = wandb.Api(timeout=20)
if api.default_entity != "jmme425":
    raise RuntimeError(
        f"W&B entity mismatch: expected jmme425, got {api.default_entity}"
    )
print("[preflight] W&B entity=", api.default_entity)
PY

ENTRYPOINT=(
    "$PYTHON_BIN" train.py
    --name Enduro-v5
    --xpid "$XPID"
    --dynamic_search True
    --wrapper_type 0
    --max_search_steps 20
    --max_depth 20
    --model_unroll_len 20
    --think_cost 0.0005
    --reward_clip 1
    --model_size_nn 2
    --discounting 0.99
    --envpool True
    --auto_res False
    --ray_cpu 12
    --ray_gpu 1
    --gpu_learn 0.5
    --gpu_learn_actor 0.25
    --gpu_self_play 0.25
    --use_wandb True
)

echo "[run] ${ENTRYPOINT[*]}"
stdbuf -oL -eL "${ENTRYPOINT[@]}"
