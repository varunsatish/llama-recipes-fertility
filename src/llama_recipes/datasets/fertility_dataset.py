# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import random
import numpy as np
import pdb
from itertools import product

def create_dataset(size):
    features = []
    outputs = []
    for _ in range(size):
        # Randomly choose feature
        if random.random() < 0.5:
            feature = "I want a child"
            output = np.random.binomial(1, 0.9)
        else:
            feature = "I don't want a child"
            output = np.random.binomial(1, 0.1)
        features.append(feature)
        outputs.append(str(output))
    return features, outputs

def create_parity_dataset(length=8):
  def calculate_parity(binary_string):
    return str(binary_string.count('1') % 2)
  # Generate all 2^8 combinations
  all_combinations = [''.join(combo) for combo in product('01', repeat=length)]
  result = [calculate_parity(combo) for combo in all_combinations]
  return all_combinations, result  

def get_preprocessed_fertility(dataset_config, tokenizer, split):
    # Create a dataset with 1000 samples
    if split == "train":
        size = 10000
        # features, outputs = create_dataset(size)
        features, outputs = create_parity_dataset(8)
    else:
        size = 500
        # features, outputs = create_dataset(size)
        features, outputs = create_parity_dataset(8)
    # Create a dataset with features and outputs, where 'dialogue' goes to features and 'summary' goes to outputs
    dataset = datasets.Dataset.from_dict({"dialogue": features, "summary": outputs})
    # dataset = datasets.load_dataset("samsum", split=split)
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        # summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"], add_special_tokens=False) # NOTE: no EOS token
        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
