#!/bin/bash

#SBATCH --nodes=1

#SBATCH --job-name=brAutoMix
#SBATCH --gres=gpu:1
#SBATCH --output=../runLogs/slurm-%j.out
#SBATCH --error=../runLogs/slurm-%j.out

source ../../venv/bin/activate

python3 test-mix-console.py