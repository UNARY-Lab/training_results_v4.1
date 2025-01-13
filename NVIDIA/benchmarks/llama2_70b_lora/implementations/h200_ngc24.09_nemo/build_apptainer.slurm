#!/bin/bash
#SBATCH --job-name=apptainer_build
#SBATCH --output=apptainer_build_%j.out
##SBATCH --error=apptainer_build_%j.err
#SBATCH -t 03:00:00
#SBATCH -p mi2104x
#SBATCH -q alloc_diwu_05142024_06302025

DIR=$SLURM_SUBMIT_DIR
# Change to the working directory
cd $DIR || { echo "Directory not found"; exit 1; }

# Build the Apptainer image
# apptainer build --sandbox --force ../nemo-container-sandbox nemo.def
apptainer build --force nemo-container.sif nemo.def

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "Apptainer image build completed successfully."
else
    echo "Apptainer image build failed. Check the logs for details."
    exit 1
fi
