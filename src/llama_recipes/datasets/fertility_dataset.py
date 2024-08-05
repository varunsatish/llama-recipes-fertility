# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import random
import numpy as np
import pdb
from itertools import product
import json

def create_dataset(size, seed, num_reps):
    features = []
    outputs = []
    rs = np.random.RandomState(seed)
    for _ in range(size):
        # Randomly choose feature
        if rs.rand() < 0.5:
            feature = "I want a child" * num_reps
            output = rs.binomial(1, 0.9)
        else:
            feature = "I don't want a child" * num_reps
            output = rs.binomial(1, 0.1)
        features.append(feature)
        outputs.append(str(output))
    return features, outputs




def create_parity_dataset(length=8):
  def calculate_parity(binary_string):
    return str(binary_string.count('1') % 2)
  # Generate all 2^length combinations
  all_combinations = [''.join(combo) for combo in product('01', repeat=length)]
  result = [calculate_parity(combo) for combo in all_combinations]
  return all_combinations, result  

def get_preprocessed_fertility(dataset_config, tokenizer, split):
    # Create a dataset with 1000 samples
    num_reps = 1   # by default, text will be repeated once. For speed tests, we will repeat num_reps times
    if dataset_config.use_parity:
        features, outputs = create_parity_dataset(8)
    else:
        size = dataset_config.train_size if split == "train" else dataset_config.valid_size
        if dataset_config.use_speed:
            num_reps = dataset_config.num_reps_if_speed
        features, outputs = create_dataset(size, seed=0 if split == "train" else 1, num_reps=num_reps)
    if dataset_config.num_extra_tokens > 0:
        features = [x + " " + " ".join(['<>'] * dataset_config.num_extra_tokens) for x in features]

    # saving dataset as json
    if dataset_config.save_dataset:
        with open(dataset_config.save_location, 'w') as file:
            json.dump({
                "text": features,
                "output": outputs}, file, indent=4)


    dataset = datasets.Dataset.from_dict({"text": features, "output": outputs})
    prompt = (
        f"Predict 1 or 0:\n{{text}}\n---\nPrediction:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(text=sample["text"]),
            "output": sample["output"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        # summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
        output = tokenizer.encode(sample["output"], add_special_tokens=False) # NOTE: no EOS token
        sample = {
            "input_ids": prompt + output,
            "attention_mask" : [1] * (len(prompt) + len(output)),
            "labels": [-100] * len(prompt) + output,
            }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
