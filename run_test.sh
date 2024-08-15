#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=benczeroAutoMix
#SBATCH --gres=gpu:1

current_time=$(date+%Y:%m:%d-%H:%M)
#SBATCH --output=slurm-%j-%x-%y_%${current_time}.out
#SBATCH --output=../runLogs/slurm-%j.out
#SBATCH --error=../runLogs/slurm-%j.out

source ../../venv/bin/activate

python3 hi.py