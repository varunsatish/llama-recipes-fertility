# writing a function that evalutates predictions over folds 

import pandas as pd
from datetime import datetime
import json
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model for sequence classification.")
    parser.add_argument("--prediction_data", type=str, required=True, help="Path to the original to obtain tokenizer")
    return parser.parse_args()

args = parse_args()

# file with predictions stored
prediction_data = pd.read_csv(args.prediction_data)
prediction_data = pd.read_csv(args.prediction_data)

### Formatting for Sklearn

# these are in the form "LABEL_0" and "LABEL_1"
predictions = prediction_data["predicted_label"]
formatted_predictons = [0 if  prediction == 'LABEL_0' else 1 for prediction in predictions]
true_labels = prediction_data["true_label"]

accuracy = accuracy_score(true_labels, formatted_predictons)
precision = precision_score(true_labels, formatted_predictons)
recall = recall_score(true_labels, formatted_predictons)
f1 = f1_score(true_labels, formatted_predictons)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)