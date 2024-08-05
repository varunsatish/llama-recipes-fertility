#!/bin/bash
#SBATCH --job-name=sweep         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --gres=gpu:4                   # total number of tasks across all nodes
#SBATCH --mem=480G         
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vs3041@princeton.edu
#SBATCH --output=sweep-%A.%a.out # stdout file
#SBATCH --error=sweep-%A.%a.err  # stderr file

module purge
module load anaconda3/2024.2

source ~/.new_recipes/bin/activate
export WANDB_MODE=offline

cd /scratch/gpfs/vs3041/cruijff/llama-recipes-fertility/recipes/quickstart/finetuning/

python hyperparameter_sweep.py
