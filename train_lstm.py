
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
from torch import nn, optim
import pickle

with open("/content/drive/MyDrive/DL/JAAD_repo/JAAD-JAAD_2.0/data_cache/trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs = np.array(self.data[idx]['obs_traj'], dtype=np.float32)  # (15, 2)
        pred = np.array(self.data[idx]['pred_traj'], dtype=np.float32)  # (45, 2)
        return torch.tensor(obs), torch.tensor(pred)

# Create dataset
dataset = TrajectoryDataset(trajectories)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2):
        super(TrajectoryLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, target_len=45):
        batch_size = input_seq.size(0)

        _, (hidden, cell) = self.encoder(input_seq)

        decoder_input = input_seq[:, -1:, :]  # last obs point: (B, 1, 2)
        outputs = []

        for _ in range(target_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)  # (B, 1, 2)
            outputs.append(pred)
            decoder_input = pred  # autoregressive

        return torch.cat(outputs, dim=1)  # (B, 45, 2)

"""# Load Model"""

model = TrajectoryLSTM().to(device)  # re-create the same model structure
model.load_state_dict(torch.load('/content/drive/MyDrive/DL/JAAD_repo/JAAD-JAAD_2.0/data_cache/trajectory_lstm_1.pth'))
model.eval()  # set to evaluation mode


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, dataloader):
    model.train()
    total_loss = 0

    for obs, pred in dataloader:
        obs = obs.to(device)  # (B, 15, 2)
        pred = pred.to(device)  # (B, 45, 2)

        optimizer.zero_grad()
        output = model(obs)  # (B, 45, 2)

        loss = criterion(output, pred)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * obs.size(0)

    return total_loss / len(dataloader.dataset)

def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for obs, pred in dataloader:
            obs = obs.to(device)
            pred = pred.to(device)

            output = model(obs)
            loss = criterion(output, pred)
            total_loss += loss.item() * obs.size(0)

    return total_loss / len(dataloader.dataset)

len(trajectories)

num_epochs = 3

for epoch in range(num_epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, test_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), './data_cache/trajectory_lstm_2.pth')
