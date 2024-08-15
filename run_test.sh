#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=brTest
#SBATCH --gres=gpu:1


#SBATCH --output=slurm-%j-${current_time}.out
#SBATCH --error=slurm-%j-${current_time}.out
current_time=$(date +%Y:%m:%d-%H:%M)
output_file="../runLogs/slurm-${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${current_time}.out"
source ../../venv/bin/activate

python3 hi.py > "$output_file" 2>&1