import subprocess
import csv
import os
from itertools import product
 
def run_experiment(batch_size, context_length, num_reps):
    command = f"""
    NAME=multi_gpu_peft
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    FSDP_CPU_RAM_EFFICIENT_LOADING=1 ACCELERATE_USE_FSDP=1 torchrun --nnodes 1 \
        --nproc_per_node 4  finetuning.py --enable_fsdp \
        --quantization 4bit \
        --model_name models/Meta-Llama-3.1-8B-Instruct \
        --mixed_precision False --low_cpu_fsdp \
        --use_peft --peft_method lora --output_dir ckpts/$NAME \
        --num_epochs 1 --run_validation True \
        --batch_size_training {batch_size} --lr 0.0003 \
        --use_fast_kernels True --context_length {context_length} \
        --batching_strategy packing --mixed_precision False --num_reps {num_reps} --train_size 1000 --valid_size 100 \
        --dataset fertility_dataset  --use_speed \
        --use-wandb --wandb_config.name $NAME
    """
 
    log_file = f"run_batchsize_{batch_size}_context_length_{context_length}_num_reps_{num_reps}.txt"
 
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                f.write(line)
                f.flush()
 
        return_code = process.wait()
        success = return_code == 0
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        success = False
 
    return success
 
def append_to_csv(batch_size, context_length, num_reps, success):
    with open('outcomes.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([batch_size, context_length, num_reps, success])
 
def main():
    batch_sizes = [32, 64]
    context_lengths = [768, 1024]
    num_reps_values = [500]
 
    # Create outcomes.csv if it doesn't exist
    if not os.path.exists('outcomes.csv'):
        with open('outcomes.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_size', 'context_length', 'num_reps', 'success'])
 
    for batch_size, context_length, num_reps in product(batch_sizes, context_lengths, num_reps_values):
        print(f"Running experiment with batch_size={batch_size}, context_length={context_length}, num_reps={num_reps}")
        success = run_experiment(batch_size, context_length, num_reps)
        append_to_csv(batch_size, context_length, num_reps, success)
        print(f"Experiment completed. Success: {success}")
 
if __name__ == "__main__":
    main()