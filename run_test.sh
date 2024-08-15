#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=benczeroAutoMix
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ../../venv/bin/activate

source ../../venv/bin/activate

python3 hi.py