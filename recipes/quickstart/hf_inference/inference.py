import wandb
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, Dataset
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, AutoPeftModelForSequenceClassification
import argparse
import pandas as pd 
from transformers import pipeline
from tqdm import tqdm
import glob
import time

def tokenize_and_prepare(data):

    # Tokenizer for HF models

    return tokenizer(data["text"], truncation=True, padding="max_length", max_length=512)

def calculate_tokens_per_second(num_tokens, elapsed_time):

    # function that calculates tokens per second

    return num_tokens / elapsed_time if elapsed_time > 0 else 0

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model for sequence classification.")
    parser.add_argument("--original_model", type=str, required=True, help="Path to the original to obtain tokenizer")
    parser.add_argument("--fine_tuned_model", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--wandb_config", type=str, required=True, help="WandB tracking")
    return parser.parse_args()

args = parse_args()


# Initialize wandb
wandb.init(project=args.wandb_config, name="inference_run")

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(args.fine_tuned_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.original_model)
tokenizer.pad_token = tokenizer.eos_token

# Create a pipeline for sequence classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# data is in HF format
test_dataset = load_from_disk(args.test_data)  



# Evaluate the model on the test set using the pipeline
prediction_probabilities = []
prediction_labels = []



total_tokens = 0
start_time = time.time()

for sample in tqdm(test_dataset):
    # Tokenize the input
    tokenized_input = tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    num_tokens = tokenized_input.input_ids.shape[1]
    total_tokens += num_tokens

    # Run inference
    prediction = classifier(sample["text"])

    label = prediction[0]["label"]
    probability = prediction[0]["score"]

    prediction_probabilities.append(probability)
    prediction_labels.append(label)

    # Calculate and log tokens per second
    elapsed_time = time.time() - start_time
    tokens_per_second = calculate_tokens_per_second(total_tokens, elapsed_time)
    
    wandb.log({
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "elapsed_time": elapsed_time
    })

# Log final metrics
wandb.log({
    "final_total_tokens": total_tokens,
    "final_tokens_per_second": calculate_tokens_per_second(total_tokens, time.time() - start_time),
    "total_elapsed_time": time.time() - start_time
})

prediction_df = pd.DataFrame({
    "probabilities": prediction_probabilities,
    "predicted_label": prediction_labels,
    "true_label": test_dataset["labels"]
})

prediction_df.to_csv(args.output_file)

# Print wandb log
print("Weights & Biases Log Summary:")
for key, value in wandb.run.summary.items():
    print(f"{key}: {value}")

# Close wandb run
wandb.finish()