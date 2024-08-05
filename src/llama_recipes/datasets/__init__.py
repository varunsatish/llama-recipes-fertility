# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from llama_recipes.datasets.fertility_dataset import get_preprocessed_fertility as get_fertility_dataset
from llama_recipes.datasets.speed_dataset import get_preprocessed_speed as get_speed_dataset
from llama_recipes.datasets.predefined_dataset import get_predefined_dataset