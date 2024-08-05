# Llama Recipes

A fork of the [Llama recipes](https://github.com/meta-llama/llama-recipes) repository. See the [original repo](https://github.com/meta-llama/llama-recipes) for more information.

## Notes before getting started
Running this script will require you to have a Hugging Face account. If you don't already, you can create one here: `https://huggingface.co/`.

## Getting started

This assumes you are able to login to either Snellius or Della using the command line. 

### Snellius

To set up the enviironment in Snellius, you first need to run this command in a terminal window:

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
```

### Della  

To set up the environment in Della, you first need to run this command in a terminal window:

```bash
conda deactivate
module load anaconda3/2024.2
```

If you want to run the fine-tuning code, you will need to ensure that you are in the `della-gpu` login node. 

## Installing requirements

### Python packages

Create a folder named `cruijff` using `mkdir cruijff`. Because we will be downloading big files, you should ensure that you locate the folder in the part of the disk that is allocated for storage. For example, in Della this would be `/scratch/gpfs/<USER>`. 

Navigate to `cruijff` and then install this repo from Github in this directory: 

```bash
cd cruijff
git clone https://github.com/varunsatish/llama-recipes-fertility
```

Install the requirements into a new virtual environment:

```bash
python3 -m venv ~/.cruijff
source ~/.cruijff/bin/activate
pip install -r llama-recipes-fertility/requirements.txt
cd llama-recipes-fertility
pip install -e .
```

Once you have created an environment and downloaded the packages, to activate the environment you only need to run:

```bash
source ~/.cruijff/bin/activate
```

### Hugging Face models

Make sure you have your Hugging Face access key ready. To obtain this, log into Hugging Face, navigate to "settings" and then "access tokens". 

```bash
huggingface-cli login
```

Follow the instructions to log in with your access key. You can select n for the question about Github access. 

In this example, we will be downloading Llama-3.1-8B-Instruct. You can use the same process to download any other model on Hugging Face. 

```bash
cd recipes/quickstart/finetuning
mkdir models/
mkdir ckpts/
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir models/Meta-Llama-3.1-8B-Instruct
```

Set wandb to offline mode:

```bash
export WANDB_MODE=offline
```


## Running the script (minimum working example)

You will first need to initialize an interactive slurm job.

```bash
salloc --nodes=1 --ntasks=1 --gres=gpu:4 --time=60:00 --mem=480G
```

Make sure to activate relevant software:

On Snellius, do
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
```

Make sure you activate the environment and navigate to the relevant directory

```bash
source ~/.cruijff/bin/activate
cd recipes/quickstart/finetuning
```

Then, run the folllowing code for a multi-GPU speed test using LoRA:

```bash
NAME=multi_gpu_peft
export CUDA_VISIBLE_DEVICES=0,1,2,3
FSDP_CPU_RAM_EFFICIENT_LOADING=1 
ACCELERATE_USE_FSDP=1 
torchrun finetuning.py \
    --nnodes 1  \
    --nproc_per_node 4 \
    --enable_fsdp  \
    --quantization 4bit \
    --model_name models/Meta-Llama-3.1-8B-Instruct  \
    --mixed_precision False 
    --low_cpu_fsdp  \
    --use_peft \
    --peft_method lora \
    --output_dir ckpts/$NAME  \
    --num_epochs 2 \
    --run_validation True  \
    --batch_size_training 1 \
    --lr 0.0003  \
    --use_fast_kernels True \
    --context_length 512  \
    --batching_strategy packing \ 
    --mixed_precision False  \
    --dataset fertility_dataset \
    --use-wandb --wandb_config.name $NAME
```


## Specifiying Custom Datasets

To specify your own dataset, save a dictionary with a list of strings named `"text"` and a list of 0s and 1s in string format named `"labels"` in json format. 

This data template illustrates a simple example of the appropriate data structure:

```json
{
    "text": [
        "I want a child",
        "I want a child",
        "I want a child",
        "I don't want a child",
        "I want a child",
        "I don't want a child"
        ], 

    "labels": [
        "0", 
        "1",
        "1",
        "0",
        "0",
        "1"
    ]


}
```

Place this dataset in `llama-recipes-fertility/recipes/quickstart/finetuning/datasets/predefined_datasets/`. When running the fine-tuning code, makesure you specify `--dataset predefined_dataset` and insert `--dataset_name <DATASET FILE NAME>`.  



## Fine-tune with PEFT on a single GPU
To train a model with PEFT, make sure you're in the `llama-recipes-fertility/recipes/quickstart/finetuning` directory and run the following command:
```bash
NAME=single_gpu_peft
export CUDA_VISIBLE_DEVICES=0
FSDP_CPU_RAM_EFFICIENT_LOADING=1 python finetuning.py  \
    --use_peft --peft_method lora --quantization 4bit  \
    --model_name models/Meta-Llama-3.1-8B-Instruct  \
    --output_dir ckpts/$NAME --num_epochs 100  \
    --run_validation True  --batch_size_training 1  \
    --lr 0.0003 --use_fast_kernels True  \
    --context_length 1024 --batching_strategy packing  \
    --mixed_precision False  --dataset fertility_dataset  \
    --use-wandb --wandb_config.name $NAME
```
This will train a model to predict fertility intentions. This took me about 95 seconds per epoch for the fertility dataset (1850 tokens/second) and 1 second per epoch on the parity dataset (on one A100 GPU). You can monitor your progress on Weights and Biases under the run `single_gpu_peft` (the link will be printed in the terminal). The `eval` tab will show the held-out loss and perplexity. The `train/tokens_per_second` tab will show you how many tokens are being processed per second (during training). If the model's heldout perplexity is 1.35-1.50 on fertility intentions I'd say the model is doing well.

Additional flags:
- `--use_parity`: fine-tune on the parity dataset instead of fertility intentions.
- `--extra_tokens N`: add `N` extra tokens to the dataset to simulate longer books of life. You may need to increase the `--context_length` flag to allow for the extra tokens.
- `--train_size N`: change the number of training examples (default: 10000). 
- `--valid_size N`: change the number of validation examples (default: 1000).

If you get an out-of-memory error try lowering the `--context_length` to something like 512. 

## Fine-tune with PEFT on multiple GPUS

```bash
NAME=multi_gpu_peft
export CUDA_VISIBLE_DEVICES=0,1,2,3
FSDP_CPU_RAM_EFFICIENT_LOADING=1 ACCELERATE_USE_FSDP=1 torchrun --nnodes 1  \
    --nproc_per_node 4  finetuning.py --enable_fsdp  \
    --quantization 4bit  \
    --model_name models/Meta-Llama-3.1-8B-Instruct  \
    --mixed_precision False --low_cpu_fsdp  \
    --use_peft --peft_method lora --output_dir ckpts/$NAME  \
    --num_epochs 100 --run_validation True  \
    --batch_size_training 1 --lr 0.0003  \
    --use_fast_kernels True --context_length 1024  \
    --batching_strategy packing --mixed_precision False  \
    --dataset fertility_dataset  \
    --use-wandb --wandb_config.name $NAME 
```
The same flags as above can be used (with the exception of `--use_parity`, which is too small for multiple GPUs). 

Note that this may not be faster than the single GPU training: it depends on whether the overhead of distributing the model across multiple GPUs is less than the speedup from training on multiple GPUs. (For me it was about as quick as single GPU training). This may change as we experiment with hyperparameters like context window size and batching. 

To try this on two GPUs, use `--nproc_per_node 2` instead of `4`. I found 4 GPUs to always be faster than 2 GPUs. 

## Full-parameter fine-tuning on multiple GPUs
To run full-parameter multi-GPU training, use the following command:
```bash
NAME=multi_gpu_full_parameter
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes 1 --nproc_per_node 4  finetuning.py \
  --enable_fsdp --model_name models/Meta-Llama-3.1-8B-Instruct \
  --dist_checkpoint_root_folder model_checkpoints \
  --output_dir ckpts/$NAME --fsdp_config.pure_bf16 \
  --use_fast_kernels --num_epochs 100 \
  --run_validation True  --batch_size_training 1 \
  --lr 0.0003 --use_fast_kernels True \
  --context_length 1024 --batching_strategy packing \
  --mixed_precision False --dataset fertility_dataset \
  --use-wandb --wandb_config.name $NAME 
```
Note that full-parameter fine-tuning can only be run on multiple GPUs -- the gradients won't fit into memory on a single GPU. You'll find full-parameter fine-tuning to be slower than the PEFT training (it took me about 4m15s per epoch). I also found that the optimization didn't work as well so we may have to experiment with hyperparameters.

## Additional information
If you want to explore the code and make some changes, the most relevant files are:
- `src/llama_recipes/datasets/fertility_dataset.py`: defining the dataset
- `src/llama_recipes/finetuning.py`: the main script for training the model

