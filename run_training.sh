#!/bin/bash
# run_training.sh - Runs the training command
# Usage: bash run_training.sh [JOBID] [NODE]

export JOBID=${1:-25363}
export NODE1=${2:-research-secure-10}

echo "Starting training on JOBID=$JOBID, NODE=$NODE1..."
echo "Using scratch dir: /scratch/$USER (mounted as /scratch in container)"

# Sync code on all nodes (head and workers) before training
echo "Syncing code to all nodes..."
srun --overlap --jobid=$JOBID --ntasks-per-node=1 bash -c 'newgrp docker << \EOF
CONTAINER_ID=$(docker ps -q -f "name=skyrl-")
if [ -n "$CONTAINER_ID" ]; then
    echo "Syncing code on $(hostname) in container $CONTAINER_ID..."
    docker exec $CONTAINER_ID bash -c "rm -rf /scratch/SkyRL && cp -r /workspace/SkyRL /scratch/SkyRL"
fi
EOF'

srun --overlap --oversubscribe --jobid=$JOBID --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --nodelist=$NODE1 bash -c 'newgrp docker << EOF
docker exec skyrl-head bash -c "source /workspace/SkyRL/api-keys.sh && source /workspace/SkyRL/env_vars.sh && cd /scratch/SkyRL/skyrl-train && bash examples/mini_swe_agent/run_mini_swe_30B.sh"
EOF'
