import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    # Load the raw data
    data = pd.read_csv(input_path)
    # Preprocess the data (e.g., normalization, noise reduction)
    processed_data = data.apply(lambda x: (x - np.mean(x)) / np.std(x))
    # Save the processed data
    processed_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data('data/raw/quantum_data.csv', 'data/processed/processed_data.csv')
