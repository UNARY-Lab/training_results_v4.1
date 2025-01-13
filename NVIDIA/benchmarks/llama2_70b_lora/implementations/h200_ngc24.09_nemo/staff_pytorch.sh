#!/bin/bash
#######################################################################
# Script to run pytorch multinode example code from:
# https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
# on DeltaAi
#######################################################################
#SBATCH --account=bcrc-dtai-gh
#SBATCH --job-name=multinode-example
#SBATCH --partition=ghx4
##SBATCH --reservation=affinity1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=ddp-mn-apptainer_%j.out
##SBATCH --error=ddp_training_%j.err

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head Node IP: $head_node_ip"

echo Node IP: $head_node_ip
echo head_node: $head_node
export LOGLEVEL=INFO

CONTAINER_HOME=/sw/user/NGC_containers
SIF_FILE=pytorch_24.09-py3.sif

echo "Job is starting on $(hostname)"

# Running the script inside the container
        # ${CONTAINER_HOME}/${SIF_FILE} \
time srun apptainer exec --nv \
        --env NCCL_DEBUG=info \
        --env SLURM_NNODES=${SLURM_NNODES} \
        --env SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE} \
        --env SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR} \
        --env head_node=${head_node} \
        --env head_node_ip=${head_node_ip} \
        nvidiasthingy.sif \
        /usr/local/bin/torchrun --nnodes ${SLURM_NNODES} \
          --nproc_per_node ${SLURM_GPUS_PER_NODE} \
          --rdzv_id $RANDOM --rdzv_backend c10d \
          --rdzv_endpoint="$head_node_ip:29500" \
          ${SLURM_SUBMIT_DIR}/multinode.py 50 10

rm -f snapshot.pt

