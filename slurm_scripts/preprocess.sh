#!/bin/bash
#SBATCH --job-name=prep_picai
#SBATCH --output=preprocess_log.out
#SBATCH --error=preprocess_error.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     
#SBATCH --mem=32G                 
#SBATCH --time=02:00:00           

# 1. Load the Anaconda module 
module load Anaconda3/2022.10

# 2. Activate environment
source activate prostate_diffusion 

# 3. Run the script
echo "Starting PI-CAI preprocessing"
python src/datasets/preprocess_picai_2d.py
echo "Preprocessing complete"
