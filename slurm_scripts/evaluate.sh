#!/bin/bash
#SBATCH --job-name=mass_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate prostate_diffusion

python scripts/mass_evaluation.py
