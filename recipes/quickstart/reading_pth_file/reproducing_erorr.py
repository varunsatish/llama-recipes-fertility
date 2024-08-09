from transformers import LlamaForCausalLM
import torch

# update your local directory
main_dir = "/scratch/gpfs/vs3041/cruijff/llama-recipes-fertility/"

model_files_uploaded_in_environment = main_dir + "recipes/quickstart/original_models/Meta-Llama-3.1-8B-Instruct/original/"

# copied from src/llama_recipes/finetuning.py with parts commented out
model = LlamaForCausalLM.from_pretrained(
        #train_config.model_name,
        model_files_uploaded_in_environment,
        #quantization_config=bnb_config,
        #use_cache=use_cache,
        use_cache=False,
        attn_implementation="sdpa", #if train_config.use_fast_kernels else None,
        device_map="auto", #if train_config.quantization and not train_config.enable_fsdp else None,
        torch_dtype=torch.float16 #if train_config.use_fp16 else torch.bfloat16,
    )
