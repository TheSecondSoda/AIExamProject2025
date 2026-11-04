import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np

class PetFinderDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, include_target=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.include_target = include_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['Id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        metadata = row.drop(['Id', 'Pawpularity'] if self.include_target else ['Id']).astype('float32').values

        if self.include_target:
            target = row['Pawpularity']
            return image, metadata, target
        else:
            return image, metadata

class NumericalDataset(Dataset):
    def __init__(self, csv_file, include_target=True):
        self.data = pd.read_csv(csv_file)
        self.include_target = include_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Drop ID and target if included
        if self.include_target:
            metadata = row.drop(['Id', 'Pawpularity']).astype('float32').values
            target = row['Pawpularity']
            return metadata, target
        else:
            metadata = row.drop(['Id']).astype('float32').values
            return metadata

# For metadata prediction only
class MetaLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.metadata_cols = [
            'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(row[self.metadata_cols].values.astype(np.float32))
        return image, target
