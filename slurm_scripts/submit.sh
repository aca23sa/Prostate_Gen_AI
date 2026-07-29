#!/bin/bash
#SBATCH --job-name=prostate_diffusion
#SBATCH --output=training_log_%j.out   
#SBATCH --error=training_error_%j.err 
#SBATCH --partition=gpu              
#SBATCH --qos=gpu                     
#SBATCH --gres=gpu:1                  
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --mem=64G                      
#SBATCH --time=12:00:00                

# 1. Initialize conda for the batch environment
source ~/.bashrc 

# 2. Activate environment
conda activate prostate_diffusion

export PYTHONPATH=$PWD
# 4. Run the script
python scripts/train_picai_monai.py
