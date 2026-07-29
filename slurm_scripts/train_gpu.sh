#!/bin/bash
#SBATCH --partition=hp-a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=diffusion_train
#SBATCH --output=logs/train_%j.out

source ~/.bashrc
conda activate prostate_diffusion

python -m scripts.train_picai_monai

