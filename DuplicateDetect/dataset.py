import os
import numpy as np
import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF

INPUT_IMAGE_SIZE = (320, 180)

class DuplicateDataset(VisionDataset):

    def __init__(self, root: str, csv_file: str):
        super().__init__(root)
        self.data = pd.read_csv(os.path.join(root, csv_file), sep=',')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        y = torch.tensor(row.iloc[2], dtype=torch.float32)
        image_paths = (row.iloc[0], row.iloc[1])
        return image_paths, y.unsqueeze(0)


def make_duplicate_collate_fn(img_server, transform=None):
    def collate_fn(batch):
        all_paths = []
        y = []
        for paths, target in batch:
            all_paths.extend(paths)
            y.append(target)

        images = img_server.ask_for_images(all_paths, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1])

        X = []
        for i in range(len(batch)):
            img1 = images[2 * i].astype(np.float32) / 255.0
            img2 = images[2 * i + 1].astype(np.float32) / 255.0
            img1 = TF.to_tensor(img1)
            img2 = TF.to_tensor(img2)
            pair = torch.cat((img1, img2), dim=0)
            if transform is not None:
                pair = transform(pair)
            X.append(pair)

        X_batch = torch.stack(X)
        y_batch = torch.stack(y)
        return X_batch, y_batch

    return collate_fn