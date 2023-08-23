#!/usr/bin/env bash

PYTHONPATH=../../../../:$PYTHONPATH

set -x

PARTITION=$1
JOB_NAME=$2
# CONFIG=$3
# CHECKPOINT=$4
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
# PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u -m prototype.solver.declip_solver --config config_ft_ours.yaml  
    #  --evaluate
    # python -u -m prototype.solver.declip_solver_adv --config config_adv.yaml  
    # --evaluate
    # python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
    



