



import numpy as np
import cv2
from sklearn.model_selection import train_test_split  # if needed
from PIL import Image
from torchvision import transforms
import os
import torch

"""#Preprocessing

##Annotation Parsing (Don't run the second cell))
"""

from jaad_data import JAAD

jaad_path = "./"
imdb = JAAD(data_path=jaad_path)

db = imdb.generate_database()  # db is a dictionary

# print(list(db.keys())[100:105])
# print(db['video_0001'])





"""###Extract the Frames (Don't run it)"""

imdb.extract_and_save_images()



"""##Building the Dataset Class"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image

class PedestrianDataset(Dataset):
    def __init__(self, trajectory_pkl_path, db_path, img_base_path, transform=None):
        # loading trajectory sequences
        with open(trajectory_pkl_path, 'rb') as f:
            trajectories = pickle.load(f)

        # loading full annotations database
        with open(db_path, 'rb') as f:
            self.db = pickle.load(f)

        self.img_base_path = img_base_path
        self.transform = transform

        # Filter out any samples whose image file is missing
        valid = []
        # todo: use all trajectories
        for sample in trajectories[:100000]:
            vid = sample['video_id']
            pid = sample['ped_id']
            frame_num = self.db[vid]['ped_annotations'][pid]['frames'][0]
            frame_name = f"{frame_num:05d}.png"
            img_path = os.path.join(self.img_base_path, f"images/{vid}/{frame_name}")
            if os.path.isfile(img_path):
                valid.append(sample)
        self.trajectories = valid

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        sample = self.trajectories[idx]

        video_id = sample['video_id']
        ped_id = sample['ped_id']
        obs_traj = sample['obs_traj']  # List of 15 (x,y) points
        pred_traj = sample['pred_traj']  # List of 45 (x,y) points

        frame_num = self.db[video_id]['ped_annotations'][ped_id]['frames'][0]
        frame_name = f"{frame_num:05d}.png"
        img_path = os.path.join(self.img_base_path, f"images/{video_id}/{frame_name}")

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Getting pedestrian attributes
        attributes = self.db[video_id]['ped_annotations'][ped_id]['attributes']

        # Creating intention label (1 if crossing, else 0)
        intention = 1 if attributes.get('crossing', False) else 0

        # Creating action label (1 if walking, else 0)
        action = 1 if attributes.get('motion', 'none') == 'walking' else 0

        # Risk label calculation
        center_x, center_y = pred_traj[-1]  # Last point of pred_traj
        region_width = 160
        img_width = 1920
        num_regions = img_width // region_width

        center_x = min(max(center_x, 0), img_width - 1)
        region_idx = int(center_x / region_width)
        risk_score = region_idx / (num_regions / 2)
        risk_score = min(risk_score, 1.0)

        # Converting to tensors
        obs_traj = torch.tensor(obs_traj, dtype=torch.float32)  # [15, 2]
        pred_traj = torch.tensor(pred_traj, dtype=torch.float32)  # [45, 2]

        labels = {
            'intention': torch.tensor(intention, dtype=torch.float32),
            'action':    torch.tensor(action,    dtype=torch.float32),
            'risk':      torch.tensor(risk_score, dtype=torch.float32)
        }

        return img, obs_traj, labels, pred_traj   #  RETURNS FUTURE TRAJECTORY

"""##Image Transformations

"""

from torchvision import transforms
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# img_path = os.path.join(jaad_path, 'images/video_0001/00000.png')
# img = Image.open(img_path).convert('RGB')
# img_tensor = transform(img)

# Image.open(img_path).convert('RGB')

"""##Initialize the Dataset
create an object dataset that knows how to load JAAD pedestrian data.
"""

dataset = PedestrianDataset(
    trajectory_pkl_path='./data_cache/trajectories.pkl',
    db_path='./data_cache/jaad_database.pkl',
    img_base_path='./',
    transform=transform
)

print("Total number of samples:", len(dataset))

"""Printing a Random Pedestrian's Behavior"""

# import random
#
# # Pick a random index
# random_idx = random.randint(0, len(dataset) - 1)
#
# # Get the sample
# img, obs_traj, labels, pred_traj = dataset[random_idx]  # Unpack correctly!
#
# print(f"Random Sample Index: {random_idx}")
# print(f"Intention (Crossing): {labels['intention'].item()}")  # 1 = Crossing, 0 = Not crossing
# print(f"Action (Walking): {labels['action'].item()}")         # 1 = Walking, 0 = Not walking
# print(f"Risk Score: {labels['risk'].item():.3f}")             # Risk (normalized 0 to 1)

"""##DataLoader"""

from torch.utils.data import random_split
import torch

# Split and save indices
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Save indices
torch.save(train_dataset.indices, './data_cache/train_indices.pt')
torch.save(test_dataset.indices, './data_cache/test_indices.pt')

from torch.utils.data import DataLoader, Subset, random_split

# Load the indices
train_indices = torch.load('./data_cache/train_indices.pt')
test_indices = torch.load('./data_cache/test_indices.pt')

# Create subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Now create your DataLoaders as before
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

"""##Reading and Checking a Batch"""

# Iterate over 1 batch
# for images, obs_traj, labels, pred_traj in train_loader:
#     print("Image batch shape:", images.shape)         # [32, 3, 224, 224]
#     print("Observed Trajectory shape:", obs_traj.shape)   # [32, 15, 2]
#     print("Intention label shape:", labels['intention'].shape) # [32]
#     print("Action label shape:", labels['action'].shape)       # [32]
#     print("Risk label shape:", labels['risk'].shape)           # [32]
#     print("Predicted Trajectory shape:", pred_traj.shape)      # [32, 45, 2]
#
#     break  # Only first batch

"""#Pretraining
Because LSTM starts randomly and needs warm-up we need to train LSTM using trajectory only and save it

"""

#small code plan:
import torch
import torch.nn as nn

# saba code
# class TrajectoryPredictor(nn.Module):
#     def __init__(self, obs_len=15, pred_len=45, hidden_size=128):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, pred_len * 2)  # Predict 45 (x,y) pairs

#     def forward(self, trajs):
#         _, (h_n, _) = self.lstm(trajs)
#         last_hidden = h_n[-1]
#         output = self.fc(last_hidden)
#         return output.view(trajs.size(0), 45, 2)  # Ensure output shape is [B, 45, 2]

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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

# mini_model = TrajectoryPredictor().to(device)
# optimizer_lstm = torch.optim.Adam(mini_model.parameters(), lr=1e-3)
# criterion_traj = nn.MSELoss()

# for epoch in range(10):
#     mini_model.train()
#     total_loss = 0.0

#     for images, obs_traj, labels, future_traj in dataloader:
#         obs_traj = obs_traj.to(device)
#         future_traj = future_traj.to(device)

#         optimizer_lstm.zero_grad()
#         pred_future_traj = mini_model(obs_traj)
#         loss = criterion_traj(pred_future_traj, future_traj)
#         loss.backward()
#         optimizer_lstm.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} - LSTM Pretrain Loss: {total_loss / len(dataloader):.4f}")

# # saving pretrained LSTM weights
# torch.save(mini_model.lstm.state_dict(), 'pretrained_lstm.pth')
# print("Pretrained LSTM weights saved as 'pretrained_lstm.pth'.")

"""## Model Definition & Forward Pass"""

import torchvision.models as models

class MultiModalPedestrianNet(nn.Module):
    def __init__(self,
                 LSTM_model,
                 pred_len: int,
                 lstm_hidden: int = 128,
                 fusion_dim: int = 512,
                 dropout_p: float = 0.3,
                 freeze_resnet: bool = True,):
        """
        pred_len: number of future frames (used only if you add the optional traj head)
        lstm_hidden: size of the LSTM hidden state
        fusion_dim: size of the joint embedding after fusion
        dropout_p: dropout probability in the fusion layer
        freeze_resnet: if True, freezes ResNet weights initially
        """
        super().__init__()
        self.pred_len = pred_len

        # ---- 1) Image branch: ResNet-50 ----
        resnet = models.resnet50(pretrained=True)
        self.resnet_fc_in = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet

        if freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

        # ---- 2) Trajectory branch: LSTM + LayerNorm ----
        # self.lstm = nn.LSTM(input_size=2,
        #                     hidden_size=lstm_hidden,
        #                     num_layers=1,
        #                     batch_first=True)
        self.lstm = LSTM_model
        self.traj_norm = nn.LayerNorm(lstm_hidden)

        # ---- 3) Fusion MLP ----
        fused_in = self.resnet_fc_in + 2*lstm_hidden
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(p=dropout_p),
        )

        # ---- 4) Task heads ----
        self.head_intent = nn.Linear(fusion_dim, 1)  # binary classification
        self.head_action = nn.Linear(fusion_dim, 1)  # binary classification
        self.head_risk   = nn.Linear(fusion_dim, 1)  # regression
        # optional future-trajectory head:
        self.head_traj   = nn.Linear(fusion_dim, pred_len * 2)

    def unfreeze_resnet(self):
        """Call this after a warm-up phase to fine-tune the ResNet weights."""
        for p in self.resnet.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor, trajs: torch.Tensor):
        """
        images: (B, 3, 224, 224)
        trajs:  (B, obs_len, 2)
        Returns:
          intent_logits: (B,1)
          action_logits:(B,1)
          risk_preds:   (B,1)
          traj_preds:   (B, pred_len, 2) — if you choose to use it
        """
        # 1) Image features
        img_feat = self.resnet(images)  # (B, 2048)

        # 2) Trajectory features
        _, (h_n, _) = self.lstm.encoder(trajs)  # h_n: (1, B, lstm_hidden)
        traj_feat = h_n.squeeze(0)      # (B, lstm_hidden)
        traj_feat = self.traj_norm(traj_feat)


        # 3) Fuse
        traj_feat_flat = traj_feat.transpose(0, 1).reshape(32, -1)  # shape: [32, 2*128] = [32, 256]
        fused = torch.cat([img_feat, traj_feat_flat], dim=1)  # (B, 2048 + 2*lstm_hidden)
        fused = self.fusion_fc(fused)                    # (B, fusion_dim)

        # 4) Heads
        intent_logits = self.head_intent(fused)           # (B,1)
        action_logits = self.head_action(fused)           # (B,1)
        risk_preds    = self.head_risk(fused)             # (B,1)

        # optional trajectory prediction
        traj_preds = self.head_traj(fused).view(images.size(0), self.pred_len, 2)

        return intent_logits, action_logits, risk_preds, traj_preds

"""##Training loop
If too few epochs → model doesn’t learn enough.

If too many epochs → model memorizes and generalizes badly.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load ONLY pretrained LSTM weights
lstm_model = TrajectoryLSTM().to(device)
lstm_weights = torch.load('./model_cache/trajectory_lstm_2.pth')
lstm_model.load_state_dict(lstm_weights)
print(" Loaded pretrained LSTM into multimodal model.")

# Initializing multimodal model
model = MultiModalPedestrianNet(lstm_model, pred_len=45).to(device)

# # Recreate your LSTM first (same structure!)
# lstm_model = TrajectoryLSTM().to(device)
#
# # Then rebuild the full model
# model = MultiModalPedestrianNet(lstm_model, pred_len=45).to(device)
#
# # Load the weights
# model.load_state_dict(torch.load('data_cache/multimodal_model.pth'))

# Optimizer & loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion_bce = nn.BCEWithLogitsLoss()
criterion_mse = nn.MSELoss()

# Training configuration
epochs = 2
print_every = 1

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, trajs, labels, future_traj) in enumerate(train_loader):
        images = images.to(device)
        trajs = trajs.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()

        # Forward pass
        intent_logits, action_logits, risk_preds, _ = model(images, trajs)

        # Compute individual losses
        loss_intent = criterion_bce(intent_logits.squeeze(), labels['intention'])
        loss_action = criterion_bce(action_logits.squeeze(), labels['action'])
        loss_risk = criterion_mse(risk_preds.squeeze(), labels['risk'])

        # Total multi-task loss
        loss = loss_intent + loss_action + loss_risk

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")

# Save full training checkpoint
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss
}
torch.save(checkpoint, './model_cache/checkpoint.pth')
print("Full checkpoint saved as 'checkpoint.pth'")

# Save model weights separately
torch.save(model.state_dict(), './model_cache/multimodal_model.pth')
print(" Final model weights saved as 'multimodal_pedestrian_model.pth'")


"""#Evaluation"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

model.eval()

all_intention_preds = []
all_intention_labels = []

all_action_preds = []
all_action_labels = []

all_risk_preds = []
all_risk_labels = []

with torch.no_grad():
    for images, obs_traj, labels, future_traj in test_loader:
        images = images.to(device)
        obs_traj = obs_traj.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        # Forward pass
        intent_logits, action_logits, risk_preds, traj_preds = model(images, obs_traj)

        # intention
        intention_pred = (torch.sigmoid(intent_logits).squeeze() > 0.5).long()
        intention_label = labels['intention'].long()

        all_intention_preds.append(intention_pred.cpu().numpy())
        all_intention_labels.append(intention_label.cpu().numpy())

        # action
        action_pred = (torch.sigmoid(action_logits).squeeze() > 0.5).long()
        action_label = labels['action'].long()

        all_action_preds.append(action_pred.cpu().numpy())
        all_action_labels.append(action_label.cpu().numpy())

        # risk
        risk_pred = risk_preds.squeeze()
        risk_label = labels['risk']

        all_risk_preds.append(risk_pred.cpu().numpy())
        all_risk_labels.append(risk_label.cpu().numpy())

all_intention_preds = np.concatenate(all_intention_preds)
all_intention_labels = np.concatenate(all_intention_labels)
all_action_preds = np.concatenate(all_action_preds)
all_action_labels = np.concatenate(all_action_labels)
all_risk_preds = np.concatenate(all_risk_preds)
all_risk_labels = np.concatenate(all_risk_labels)

intention_acc = accuracy_score(all_intention_labels, all_intention_preds)
action_acc = accuracy_score(all_action_labels, all_action_preds)
risk_rmse = np.sqrt(mean_squared_error(all_risk_labels, all_risk_preds))

print("\n==== Evaluation Results ====")
print(f"Intention Accuracy: {intention_acc*100:.2f}%")
print(f"Action Accuracy:    {action_acc*100:.2f}%")
print(f"Risk RMSE:          {risk_rmse:.4f}")

""":==== Evaluation Results ====
Intention Accuracy: 85.23%
Action Accuracy:    81.67%
Risk RMSE:          0.1523

"""

