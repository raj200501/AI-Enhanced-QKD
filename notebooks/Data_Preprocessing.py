# Data_Preprocessing.ipynb

import pandas as pd
import numpy as np

# Load raw data
data = pd.read_csv('../data/raw/quantum_data.csv')

# Preprocess data (e.g., normalization)
processed_data = data.apply(lambda x: (x - np.mean(x)) / np.std(x))

# Save processed data
processed_data.to_csv('../data/processed/processed_data.csv', index=False)

# Split data into training, validation, and test sets
train_data = processed_data.sample(frac=0.7, random_state=42)
temp_data = processed_data.drop(train_data.index)
val_data = temp_data.sample(frac=0.5, random_state=42)
test_data = temp_data.drop(val_data.index)

# Save split data
np.save('../data/processed/train_data.npy', train_data.drop(columns='label').values)
np.save('../data/processed/train_labels.npy', train_data['label'].values)
np.save('../data/processed/val_data.npy', val_data.drop(columns='label').values)
np.save('../data/processed/val_labels.npy', val_data['label'].values)
np.save('../data/processed/test_data.npy', test_data.drop(columns='label').values)
np.save('../data/processed/test_labels.npy', test_data['label'].values)
