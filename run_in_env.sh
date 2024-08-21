#!/bin/bash

#SBATCH --nodes=1

#SBATCH --job-name=brAutoMix
#SBATCH --gres=gpu:1 -w Lenovo2
current_time=$(date +%Y:%m:%d-%H:%M)
output_file="../runLogs/slurm-${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${current_time}.out"

source ../../venv/bin/activate

python3 test-mix-console.py > $output_file 2>&1