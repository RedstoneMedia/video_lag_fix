import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, Subset
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF
from img_process_server_connect import ImageProcessServerConnect

INPUT_IMAGE_SIZE = (320, 180)

class MotionDataset(VisionDataset):

    def __init__(self, csv_file: str, img_server: ImageProcessServerConnect, transform=None):
        super().__init__(img_server.working_dir, transform=transform)
        self.data = pd.read_csv(os.path.join(img_server.working_dir, csv_file), sep=',')
        self.transform = transform
        self.img_server = img_server

        all_targets = self.data.apply(self.calc_raw_target_row, axis=1).values
        self.y_mean = all_targets.mean()
        self.y_std = all_targets.std()


    @staticmethod
    def calc_raw_target(motion: float, ax: float, ay: float) -> float:
        motion2 = math.sqrt(math.pow(ax, 2) + math.pow(ay, 2))
        if math.isclose(motion, 0.0, abs_tol=1e-7):
            return 0.0
        return 0.8 * motion + 0.2 * motion2

    @staticmethod
    def calc_raw_target_row(row) -> float:
        return MotionDataset.calc_raw_target(float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_paths = [row.iloc[0], row.iloc[1]]
        target = self.calc_raw_target_row(row)
        target = (target - self.y_mean) / self.y_std # Norm
        images = self.img_server.ask_for_images(image_paths, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1])
        img1 = images[0]
        img2 = images[1]

        norm1 = img1.astype(np.float32) / 255.0
        norm2 = img2.astype(np.float32) / 255.0
        diff = TF.to_tensor(norm1 - norm2).to(torch.float32)

        if self.transform:
            diff = self.transform(diff)

        return diff, torch.tensor(target, dtype=torch.float32)

