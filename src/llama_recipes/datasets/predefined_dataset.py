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

# predefined datasets should come in dictionary format with "text" and "labels"

def get_predefined_dataset(dataset_config, tokenizer, split):

    # should be in json format
    file_path = "datasets/predefined_datasets/" + dataset_config.dataset_name

    with open(file_path, 'r') as file:
        data_dict = json.load(file)

    print("The columns in the dataset are: " + str(list(data_dict.keys())))

    if "text" not in list(data_dict.keys()):
        raise KeyError("The 'text' key is missing from the dictionary")
    if "labels" not in list(data_dict.keys()):
        raise KeyError("The 'labels' key is missing from the dictionary")

    dataset = datasets.Dataset.from_dict(data_dict)

    prompt = (
        f"Predict 1 or 0:\n{{text}}\n---\nPrediction:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(text=sample["text"]),
            "labels": sample["labels"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        # summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
        output = tokenizer.encode(sample["labels"], add_special_tokens=False) # NOTE: no EOS token

        sample = {
            "input_ids": prompt + output,
            "attention_mask" : [1] * (len(prompt) + len(output)),
            "labels": [-100] * len(prompt) + output,
            }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
