import numpy as np
import cv2
from sklearn.model_selection import train_test_split  # if needed
from PIL import Image
from torchvision import transforms
import os
import torch
import pickle
from torch.utils.data import Dataset
from jaad_data import JAAD
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, mean_squared_error
import torch.nn as nn
import torchvision.models as models
import seaborn as sns



"""#Preprocessing

##Annotation Parsing (Don't run the second cell))
"""



jaad_path = "./"
imdb = JAAD(data_path=jaad_path)

"""##Load Parsed Data"""


pkl_path = "./data_cache/jaad_database.pkl"

with open(pkl_path, "rb") as f:
    db = pickle.load(f)


"""##Building the Dataset Class"""



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

        return img, obs_traj, labels, pred_traj, video_id, ped_id   #  RETURNS FUTURE TRAJECTORY

"""##Image Transformations

"""



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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




"""##DataLoader"""


# Load the indices
train_indices = torch.load('./data_cache/train_indices.pt')
test_indices = torch.load('./data_cache/test_indices.pt')

# Create subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Now create your DataLoaders as before
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)


"""#Pretraining
Because LSTM starts randomly and needs warm-up we need to train LSTM using trajectory only and save it

"""


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


"""## Model Definition & Forward Pass"""

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
        traj_feat_flat = traj_feat.transpose(0, 1).reshape(img_feat.size()[0], -1)  # shape: [32, 2*128] = [32, 256]
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
# lstm_model = TrajectoryLSTM().to(device)
# lstm_weights = torch.load('./model_cache/trajectory_lstm_2.pth')
# lstm_model.load_state_dict(lstm_weights)
# print(" Loaded pretrained LSTM into multimodal model.")

# # Initializing multimodal model
# model = MultiModalPedestrianNet(lstm_model, pred_len=45).to(device)


lstm_model = TrajectoryLSTM().to(device)

# Then rebuild the full model
model = MultiModalPedestrianNet(lstm_model, pred_len=45).to(device)

# Load the weights
model.load_state_dict(torch.load('./model_cache/multimodal_model.pth'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

def evaluate_model(model, dataset, batch_size=32, device='cpu'):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_results = []

    with torch.no_grad():
        for imgs, trajs, labels, fut_trajs, vids, pids in tqdm(loader):
            imgs = imgs.to(device)
            trajs = trajs.to(device)

            intent_logits, action_logits, risk_preds, traj_preds = model(imgs, trajs)

            y_int = labels['intention'].to(device)
            y_act = labels['action'].to(device)
            y_risk = labels['risk'].to(device)
            y_traj = fut_trajs.to(device)

            intent_preds = (torch.sigmoid(intent_logits) > 0.5).float()
            action_preds = (torch.sigmoid(action_logits) > 0.5).float()

            for i in range(len(vids)):
                vid = vids[i]
                pid = pids[i]

                video_data = dataset.db.get(vid, {})
                ped_data = video_data.get('ped_annotations', {}).get(pid, {})
                attributes = ped_data.get('attributes', {})
                traffic_data = video_data.get('traffic_annotations', {})

                # Extract scenario elements
                motion = attributes.get('motion', 'unknown')
                crossing = attributes.get('crossing', False)
                age = attributes.get('age', 'unknown')
                gender = attributes.get('gender', 'unknown')
                road_type = traffic_data.get('road_type', 'unknown')

                # Scenario tag (e.g. walking_True_adult_male_crosswalk)
                scenario_str = f"{motion}_{crossing}_{age}_{gender}_{road_type}"

                result = {
                    "video_id": vid,
                    "ped_id": pid,
                    "intent_gt": y_int[i].item(),
                    "intent_pred": intent_preds[i].item(),
                    "action_gt": y_act[i].item(),
                    "action_pred": action_preds[i].item(),
                    "risk_gt": y_risk[i].item(),
                    "risk_pred": risk_preds[i].item(),
                    "traj_mse": F.mse_loss(traj_preds[i], y_traj[i]).item(),
                    "motion": motion,
                    "crossing": crossing,
                    "age": age,
                    "gender": gender,
                    "road_type": road_type,
                    "scenario": scenario_str
                }
                all_results.append(result)

    return pd.DataFrame(all_results)


def compute_metrics(df):
    print("=== Global Metrics ===")
    for target in ['intent', 'action']:
        acc = accuracy_score(df[f"{target}_gt"], df[f"{target}_pred"])
        bacc = balanced_accuracy_score(df[f"{target}_gt"], df[f"{target}_pred"])
        print(f"{target.capitalize()} Accuracy: {acc:.3f}, Balanced Accuracy: {bacc:.3f}")

    print(f"Risk MSE: {mean_squared_error(df['risk_gt'], df['risk_pred']):.3f}")
    print(f"Trajectory MSE: {df['traj_mse'].mean():.3f}")


def scenario_analysis(df, tag='scenario'):
    print(f"\n=== Scenario-based Metrics by {tag} ===")
    for value in df[tag].unique():
        sub = df[df[tag] == value]
        if len(sub) < 10:
            continue
        acc = accuracy_score(sub['intent_gt'], sub['intent_pred'])
        bacc = balanced_accuracy_score(sub['intent_gt'], sub['intent_pred'])
        traj_mse = sub['traj_mse'].mean()
        print(f"{value:40}: Acc={acc:.3f}, BAcc={bacc:.3f}, Traj MSE={traj_mse:.3f}")


def plot_performance(df, tag='scenario'):
    top_tags = df[tag].value_counts().nlargest(10).index
    df_plot = df[df[tag].isin(top_tags)]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_plot, x=tag, y='traj_mse', ci='sd')
    plt.title(f"Trajectory Error across Top {tag} Values")
    plt.ylabel("Trajectory MSE")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Example usage
# model = YourModel().to('cpu')
# dataset = PedestrianDataset(...)

results_df = evaluate_model(model, dataset, batch_size=32, device='cpu')

results_df.to_csv('./data_cache/scenario_results.csv', index=False)

compute_metrics(results_df)
scenario_analysis(results_df, tag='scenario')  # or 'road_type', 'motion', etc.
plot_performance(results_df, tag='scenario')
