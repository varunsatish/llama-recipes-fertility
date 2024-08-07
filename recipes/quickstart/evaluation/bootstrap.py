import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model for sequence classification.")
    parser.add_argument("--prediction_data", type=str, required=True, help="Path to the original to obtain tokenizer")
    return parser.parse_args()

args = parse_args()

def bootstrap_f1_scores(df, n_bootstrap=1000):
    # Calculate the threshold c
    c = 1 - df['true_label'].mean()
    
    probabilities = df['probabilities'].values
    true_labels = df['true_label'].values
    n_samples = len(probabilities)
    
    f1_scores = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Bootstrap the probabilities
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_probs = probabilities[boot_indices]
        boot_true = true_labels[boot_indices]
        
        # Generate predictions
        boot_preds = (boot_probs > c).astype(int)
        
        # Calculate F1 score
        f1_scores[i] = f1_score(boot_true, boot_preds)
    
    return f1_scores, c

# Read the CSV file
df = pd.read_csv(args.prediction_data)

# Perform bootstrap and calculate F1 scores
f1_scores, threshold = bootstrap_f1_scores(df)

# Calculate statistics
mean_f1 = np.mean(f1_scores)
median_f1 = np.median(f1_scores)
ci_lower = np.percentile(f1_scores, 2.5)
ci_upper = np.percentile(f1_scores, 97.5)

# Print results
print(f"Threshold c: {threshold}")
print(f"Mean F1 Score: {mean_f1:.4f}")
print(f"Median F1 Score: {median_f1:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
