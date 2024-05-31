import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cnn_anomaly_detection import create_cnn_model
from rnn_error_correction import RNNModel

def evaluate_cnn_model(model_path, test_data, test_labels):
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    return accuracy

def evaluate_rnn_model(model_path, test_data, test_labels, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_data, test_labels = torch.tensor(test_data, dtype=torch.float32).to(device), torch.tensor(test_labels, dtype=torch.float32).to(device)
    outputs = model(test_data)
    loss = torch.nn.MSELoss()(outputs, test_labels)
    print(f'Test Loss: {loss.item()}')
    return loss.item()

def plot_results(history, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.savefig(f'{save_path}/accuracy.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.savefig(f'{save_path}/loss.png')
    plt.close()

if __name__ == "__main__":
    # Load test data
    test_data = np.load('data/processed/test_data.npy')
    test_labels = np.load('data/processed/test_labels.npy')
    # Evaluate CNN model
    cnn_accuracy = evaluate_cnn_model('models/cnn_anomaly_detection/model.h5', test_data, test_labels)
    # Load RNN config
    import json
    with open('config/rnn_config.json', 'r') as f:
        rnn_config = json.load(f)
    # Evaluate RNN model
    rnn_loss = evaluate_rnn_model('models/rnn_error_correction/model.pth', test_data, test_labels, rnn_config)
    # Plot results (assuming you have saved the training history as history object)
    # plot_results(history, 'results/figures/')
