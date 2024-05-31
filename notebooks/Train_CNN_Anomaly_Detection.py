# Train_CNN_Anomaly_Detection.ipynb

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load preprocessed data
train_data = np.load('../data/processed/train_data.npy')
train_labels = np.load('../data/processed/train_labels.npy')
val_data = np.load('../data/processed/val_data.npy')
val_labels = np.load('../data/processed/val_labels.npy')

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Load configuration
config = {
    "input_shape": (64, 64, 1),
    "epochs": 20
}

# Train the CNN model
model = create_cnn_model(config['input_shape'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=config['epochs'], validation_data=(val_data, val_labels))

# Save the model
model.save('../models/cnn_anomaly_detection/model.h5')
