from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import os
import json
from collections import OrderedDict

# # ### New code:
model_path = "original_models/Meta-Llama-3.1-8B-Instruct/"

# config = AutoConfig.from_pretrained(model_path, config_file="config.json")
# model = AutoModelForCausalLM.from_config(config)

# state_dict = torch.load(model_path + "original/consolidated.00.pth", map_location="cpu")

# def rename_key(key):
#     if key.startswith('tok_embeddings'):
#         return key.replace('tok_embeddings', 'model.embed_tokens')
#     elif key.startswith('layers'):
#         parts = key.split('.')
#         layer_num = parts[1]
#         if 'attention' in key and 'norm' not in key:
#             if 'wq' in key:
#                 return f'model.layers.{layer_num}.self_attn.q_proj.weight'
#             elif 'wk' in key:
#                 return f'model.layers.{layer_num}.self_attn.k_proj.weight'
#             elif 'wv' in key:
#                 return f'model.layers.{layer_num}.self_attn.v_proj.weight'
#             elif 'wo' in key:
#                 return f'model.layers.{layer_num}.self_attn.o_proj.weight'
#         elif 'feed_forward' in key:
#             if 'w1' in key:
#                 return f'model.layers.{layer_num}.mlp.gate_proj.weight'
#             elif 'w2' in key:
#                 return f'model.layers.{layer_num}.mlp.down_proj.weight'
#             elif 'w3' in key:
#                 return f'model.layers.{layer_num}.mlp.up_proj.weight'
#         elif 'attention_norm' in key:
#           return f'model.layers.{layer_num}.input_layernorm.weight'
#         elif 'ffn_norm' in key:
#             return f'model.layers.{layer_num}.post_attention_layernorm.weight'
#     elif key == 'norm.weight':
#         return 'model.norm.weight'
#     elif key == 'output.weight':
#         return 'lm_head.weight'
#     return key

# new_state_dict = OrderedDict((rename_key(k), v) for k, v in state_dict.items())
# model.load_state_dict(new_state_dict, strict=False)

# print("model loaded!")

# Make sure model is generating reasonable sequences
tokenizer = AutoTokenizer.from_pretrained(model_path + "original/", use_fast=False)
prompt = "Hey, are you conscious? Can you talk to me?"

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])





