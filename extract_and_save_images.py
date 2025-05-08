



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
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

"""#Preprocessing

##Annotation Parsing (Don't run the second cell))
"""



jaad_path = "./"
imdb = JAAD(data_path=jaad_path)

db = imdb.generate_database()  # db is a dictionary



"""###Extract the Frames (Don't run it)"""

imdb.extract_and_save_images()



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


"""##DataLoader"""



# Split and save indices
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Save indices
torch.save(train_dataset.indices, './data_cache/train_indices.pt')
torch.save(test_dataset.indices, './data_cache/test_indices.pt')



# Now create your DataLoaders as before
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

