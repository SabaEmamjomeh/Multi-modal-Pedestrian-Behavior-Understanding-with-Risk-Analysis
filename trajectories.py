
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

"""##Load Parsed Data"""

import pickle

pkl_path = "./data_cache/jaad_database.pkl"

with open(pkl_path, "rb") as f:
    db = pickle.load(f)

# print(list(db.keys())[100:105])
# print(db['video_0001'])

"""##Building the Dataset Class"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image

import pickle

def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w/2, y + h)

obs_len = 15
pred_len = 45
stride = 1  # To reduce overlap and improve diversity we can increase this number
trajectories = []

for vid in db:
    video_data = db[vid]
    ped_annotations = video_data['ped_annotations']

    for ped_id in ped_annotations:
        ped_data = ped_annotations[ped_id]
        frames = ped_data['frames']
        bboxes = ped_data['bbox']

        if len(frames) >= obs_len + pred_len and len(frames) == len(bboxes):
            for i in range(0, len(frames) - obs_len - pred_len + 1, stride):
                obs_traj = [bbox_center(b) for b in bboxes[i:i+obs_len]]
                pred_traj = [bbox_center(b) for b in bboxes[i+obs_len:i+obs_len+pred_len]]

                trajectories.append({
                    'video_id': vid,
                    'ped_id': ped_id,
                    'obs_traj': obs_traj,
                    'pred_traj': pred_traj
                })

# Save the output
save_path = "./data_cache/trajectories.pkl"

with open(save_path, 'wb') as f:
    pickle.dump(trajectories, f)

print(f"Total trajectories: {len(trajectories)}")
print(trajectories[0])

