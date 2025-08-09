import os
import pandas as pd
from PIL import Image
import torch
from PIL.Image import Resampling
from torch.utils.data import random_split, Subset
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF

INPUT_IMAGE_SIZE = (320, 180)

class MotionDataset(VisionDataset):
    def __init__(self, root: str, csv_file: str, transform=None):
        super().__init__(root, transform=transform)
        self.data = pd.read_csv(os.path.join(root, csv_file), sep=',')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img1_path = os.path.join(self.root, row.iloc[0])
        img2_path = os.path.join(self.root, row.iloc[1])
        target = float(row.iloc[2])

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        img1 = TF.to_tensor(img1).to(torch.float32)
        img2 = TF.to_tensor(img2).to(torch.float32)

        diff = img1 - img2

        if self.transform:
            diff = self.transform(diff)

        return diff, torch.tensor(target, dtype=torch.float32)


def get_split_train_val(val_split: float, root: str, csv_file: str, *args, **kwargs) -> list[Subset[MotionDataset]]:
    dataset = MotionDataset(root, csv_file, *args, **kwargs)
    generator = torch.Generator().manual_seed(42)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=generator)
