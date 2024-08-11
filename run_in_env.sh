#!/bin/bash

#SBATCH --nodes=1

#SBATCH --job-name=benczeroAutoMix

#SBATCH --gres=gpu:1

source ../../venv/bin/activate

python3 test-mix-console.py