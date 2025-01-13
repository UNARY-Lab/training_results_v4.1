#!/bin/bash
#######################################################################
# Marco's attempt to run NeMo on DeltaAI
#######################################################################
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=nemo-example
#SBATCH -t 00:30:00
#SBATCH -p mi3008x
#SBATCH -q alloc_diwu_05142024_06302025
#SBATCH --output=nemo-apptainer_%j.out
##SBATCH --error=nemo-apptainer_%j.err

set -ex

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node=localhost
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
head_node_ip=127.0.0.1
echo "Head Node IP: $head_node_ip"
export APPTAINERENV_MASTER_ADDR=127.0.0.1
export APPTAINERENV_MASTER_PORT=29521

echo Node IP: $head_node_ip
echo head_node: $head_node
export LOGLEVEL=INFO

CONTAINER_HOME=$SLURM_SUBMIT_DIR
SIF_FILE=nemo-container.sif

echo "Job is starting on $(hostname)"

APPTAINER_RUN=("apptainer" "run" "--rocm" "--bind" "$SLURM_SUBMIT_DIR/../:/data" "--env" "HF_HOME=/tmp/mkurzynski/huggingface" "${CONTAINER_HOME}/${SIF_FILE}")

# No longer needed:
# ${APPTAINER_RUN[@]} python ${SLURM_SUBMIT_DIR}/scripts/download_model.py --model_dir /data/model --hf_token ${MY_HF_TOKEN}
# 
# ${APPTAINER_RUN[@]} python ${SLURM_SUBMIT_DIR}/scripts/download_dataset.py --data_dir /data/gov_report
# ${APPTAINER_RUN[@]} python ${SLURM_SUBMIT_DIR}/scripts/convert_dataset.py --data_dir /data/gov_report
# ${APPTAINER_RUN[@]} python ${SLURM_SUBMIT_DIR}/scripts/convert_to_fused.py --model_dir /data/fused --hf_token ${MY_HF_TOKEN}
# ${APPTAINER_RUN[@]} python ${SLURM_SUBMIT_DIR}/scripts/convert_model.py --input_name_or_path=/data/fused/llama2-7b --output_path=/data/fused/llama2-7b/llama2-7b.nemo
# ${APPTAINER_RUN[@]} bash -c "cd /data/fused/llama2-7b && find . -type f ! -name 'llama2-7b.nemo' -exec rm -f {} + && tar -xvf llama2-7b.nemo"

source ${SLURM_SUBMIT_DIR}/config_DGXH200_1x8x2xtp1pp1cp2.sh
# Running the script inside the container
echo "RUNANDTIME_START $(date +%s)"
time srun apptainer exec --rocm \
        --env NCCL_DEBUG=info \
        --env SLURM_NNODES=1 \
        --env SLURM_GPUS_PER_NODE=8 \
        --env SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR} \
        --env head_node=${head_node} \
        --env head_node_ip=${head_node_ip} \
        --bind $SLURM_SUBMIT_DIR/../gov_report:/data \
        --bind $SLURM_SUBMIT_DIR/../fused/llama2-7b:/ckpt \
        ${CONTAINER_HOME}/${SIF_FILE} \
        torchrun --nnodes 1 \
          --nproc_per_node 8 \
          ${SLURM_SUBMIT_DIR}/multinode.py 50 10
          # train.py
echo "RUNANDTIME_STOP $(date +%s)"

# time srun apptainer exec --nv \
#         --env NCCL_DEBUG=info \
#         --env SLURM_NNODES=${SLURM_NNODES} \
#         --env SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE} \
#         --env SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR} \
#         --env head_node=${head_node} \
#         --env head_node_ip=${head_node_ip} \
#         --bind $SLURM_SUBMIT_DIR/../:/data \
#         --bind $SLURM_SUBMIT_DIR/../fused/llama2-7b:/ckpt \
#         ${CONTAINER_HOME}/${SIF_FILE} \
#          bash -c "ls /usr/lib64/"
rm -f snapshot.pt

