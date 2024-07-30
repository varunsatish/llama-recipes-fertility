import random
import numpy as np

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
        outputs.append(output)
    return features, outputs

# Create a dataset with 1000 samples
size = 1000
features, outputs = create_dataset(size)

# Print the first 10 samples
for i in range(10):
    print(f"Feature: {features[i]}, Output: {outputs[i]}")

# Print dataset statistics
want_child = features.count("i want a child")
dont_want_child = features.count("i don't want a child")
positive_outputs = outputs.count(1)

print(f"\nDataset statistics:")
print(f"Total samples: {size}")
print(f"'i want a child': {want_child}")
print(f"'i don't want a child': {dont_want_child}")
print(f"Positive outputs: {positive_outputs}")