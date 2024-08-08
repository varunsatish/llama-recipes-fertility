#!/bin/bash

#SBATCH --job-name=experiment_1
#SBATCH -p gpu
#SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -e train_inf_output/exp_1.err 
#SBATCH -o train_inf_output/exp_1.out 

export WANDB_MODE=offline

export ROOT_DIR=/home/fhafner/
export REPO_DIR=$ROOT_DIR/repositories/llama-recipes-fertility
export VENV=$REPO_DIR/.venv/bin/activate

#load the modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

#source the virtual environment
source $VENV

# Move to location
cd $REPO_DIR/recipes/quickstart/

NAME=multi_gpu_peft
export CUDA_VISIBLE_DEVICES=0,1,2,3
FSDP_CPU_RAM_EFFICIENT_LOADING=1 ACCELERATE_USE_FSDP=1 torchrun --nnodes 1  \
    --nproc_per_node 4  finetuning/finetuning.py --enable_fsdp  \
    --quantization 4bit  \
    --model_name original_models/Meta-Llama-3.1-8B-Instruct  \
    --mixed_precision False --low_cpu_fsdp  \
    --use_peft --peft_method lora --output_dir train_inf_output/$NAME  \
    --num_epochs 2 --run_validation True  \
    --batch_size_training 1 --lr 0.0003  \
    --use_fast_kernels True --context_length 512  \
    --batching_strategy packing --mixed_precision False  \
    --dataset minimum_working_example  \
    --use-wandb --wandb_config.name $NAME \
    --dist_backend nccl # don't use gloo unless for testing! 








