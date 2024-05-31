# Train_RNN_Error_Correction.ipynb

import torch
import torch.nn as nn
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(config['num_layers'], x.size(0), config['hidden_size']).to(device) 
        c_0 = torch.zeros(config['num_layers'], x.size(0), config['hidden_size']).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Load preprocessed data
train_data = np.load('../data/processed/train_data.npy')
train_labels = np.load('../data/processed/train_labels.npy')
val_data = np.load('../data/processed/val_data.npy')
val_labels = np.load('../data/processed/val_labels.npy')

# Load configuration
config = {
    "input_size": 64,
    "hidden_size": 64,
    "output_size": 1,
    "num_layers": 2,
    "epochs": 20,
    "learning_rate": 0.001
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers']).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

train_data, train_labels = torch.tensor(train_data, dtype=torch.float32).to(device), torch.tensor(train_labels, dtype=torch.float32).to(device)
val_data, val_labels = torch.tensor(val_data, dtype=torch.float32).to(device), torch.tensor(val_labels, dtype=torch.float32).to(device)

for epoch in range(config['epochs']):
    model.train()
    outputs = model(train_data)
    optimizer.zero_grad()
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    
    model.eval()
    val_outputs = model(val_data)
    val_loss = criterion(val_outputs, val_labels)
    print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

torch.save(model.state_dict(), '../models/rnn_error_correction/model.pth')
