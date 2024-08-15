#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=brTest
#SBATCH --gres=gpu:1

current_time=$(date +%Y:%m:%d-%H:%M)
#SBATCH --output=../runLogs/slurm-%j-${current_time}.out
#SBATCH --error=../runLogs/slurm-%j-${current_time}.out

source ../../venv/bin/activate

python3 hi.py