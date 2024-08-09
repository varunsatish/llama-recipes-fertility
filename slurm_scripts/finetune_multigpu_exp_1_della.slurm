#!/bin/bash
#SBATCH --job-name=exp_1    # create a short name for your job
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=200G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH -o output_from_slurm/exp_1.out # output file 

export WANDB_MODE=offline

export ROOT_DIR=/scratch/gpfs/vs3041/
export REPO_DIR=$ROOT_DIR/cruijff/llama-recipes-fertility
export VENV=$REPO_DIR/.venv/bin/activate

#load the modules
module load anaconda3/2024.2

#source the virtual environment
source $VENV

# Move to location
cd $REPO_DIR/recipes/quickstart/

NAME=multi_gpu_peft
export CUDA_VISIBLE_DEVICES=0,1
FSDP_CPU_RAM_EFFICIENT_LOADING=1 ACCELERATE_USE_FSDP=1 torchrun --nnodes 1  \
    --nproc_per_node 2  finetuning/finetuning.py --enable_fsdp  \
    --quantization 4bit  \
    --model_name original_models/Meta-Llama-3.1-8B-Instruct  \
    --mixed_precision False --low_cpu_fsdp  \
    --use_peft --peft_method lora --output_dir train_inf_output/$NAME  \
    --num_epochs 2 --run_validation True  \
    --batch_size_training 1 --lr 0.0003  \
    --use_fast_kernels True --context_length 512  \
    --batching_strategy packing --mixed_precision False  \
    --dataset predefined_dataset  \
    --data_path finetuning/datasets/predefined_datasets/testing_BOL \
    --use-wandb 







