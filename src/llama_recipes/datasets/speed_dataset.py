# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import random
import numpy as np
import pdb
from itertools import product
import string


def generate_random_text(tokens_per_sample):

    # Generate a large chunk of random text
    chunk_size = tokens_per_sample * 4  # Estimating 4 characters per token on average
    random_text = "".join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=chunk_size))

    return random_text

def create_dataset(seed, tokens_per_sample, size):

    # set seed 
    rs = random.seed(seed)

    features = generate_random_text(tokens_per_sample=tokens_per_sample)

    outputs = [random.choice([0, 1]) for _ in range(size)]
    return features, outputs


def get_preprocessed_speed(dataset_config, tokenizer, split):
    size = dataset_config.train_size if split == "train" else dataset_config.valid_size
    features, outputs = create_dataset(size, seed=0 if split == "train" else 1)
    if dataset_config.num_extra_tokens > 0:
        features = [x + " " + " ".join(['<>'] * dataset_config.num_extra_tokens) for x in features]
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
