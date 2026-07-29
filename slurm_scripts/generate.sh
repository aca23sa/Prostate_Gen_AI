#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G

module load Anaconda3/2022.10
source activate prostate_diffusion

python scripts/generate_conditional.py
