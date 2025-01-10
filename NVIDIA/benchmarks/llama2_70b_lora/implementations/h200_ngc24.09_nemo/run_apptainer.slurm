#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=nemo_apptainer_test
#SBATCH --output=apptainer_run%j.out
##SBATCH --error=apptainer_run%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ghx4
#SBATCH --account=bcrc-dtai-gh

set -ex

DIR=/work/nvme/bcrc/mkurzynski/github/training_results_v4.1/NVIDIA/benchmarks/llama2_70b_lora/implementations/h200_ngc24.09_nemo
# Change to the working directory
cd $DIR || { echo "Directory not found"; exit 1; }

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
# export RDZV_HOST=$(hostname)
# export RDZV_PORT=42070
# export MASTER_PORT="$RDZV_PORT"
# export MASTER_ADDR="$RDZV_HOST"

if [ "${node_rank:-0}" -eq 0 ] && [ "${local_rank:-0}" -eq 0 ]
then
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
fi

APPTAINER_RUN=("apptainer" "run" "--nv" "--bind" "$DIR/../:/data" "nemo-container.sif")

# ${APPTAINER_RUN[@]} python scripts/download_dataset.py --data_dir /data/gov_report
# ${APPTAINER_RUN[@]} python scripts/download_model.py --model_dir /data/model
# ${APPTAINER_RUN[@]} python scripts/convert_dataset.py --data_dir /data/gov_report
# ${APPTAINER_RUN[@]} python scripts/convert_model.py --input_name_or_path=/data/model --output_path=/data/model/llama2-70b.nemo
# ${APPTAINER_RUN[@]} bash -c "cd /data/model && find . -type f ! -name 'llama2-70b.nemo' -exec rm -f {} + && tar -xvf llama2-70b.nemo"

source $DIR/config_DGXH200_1x8x2xtp1pp1cp2.sh
export HYDRA_FULL_ERROR=1
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')

export LOGLEVEL=INFO

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load nccl

export APPTAINERENV_MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export APPTAINERENV_MASTER_PORT=29521
APPTAINER_RUN=("apptainer" "exec" "--nv" "--bind" "$DIR/../:/data" "--bind" "$DIR/../model/:/ckpt" "nemo-container.sif")
# CMD=( 'torchrun' '--nnodes=1' '--nproc_per_node=4' '--rdzv_id' "$RANDOM" '--rdzv_backend' 'c10d' "--rdzv_endpoint=$head_node_ip:29500" )
CMD=( 'torchrun' '--nproc_per_node=4' )
${APPTAINER_RUN[@]} ${CMD[@]} train.py; ret_code=$?
# srun --mpi=pmi2 ${APPTAINER_RUN[@]} ${CMD[@]} train.py; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
    # report result
    result=$(( $end - $start ))
    result_name="LLM_FINETUNING"
    echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
fi
