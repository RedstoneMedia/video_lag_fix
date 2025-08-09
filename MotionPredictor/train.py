import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data import DataLoader

from torchinfo import summary
from safetensors.torch import save_file

from dataset import INPUT_IMAGE_SIZE, get_split_train_val


DATA_ROOT = r"E:\Download\Data\ml\motion_predictor"

class TinyMotionNet(nn.Module):

    def __init__(self):
        super().__init__()
        channels = [8, 16, 32, 64]
        prev_ch  = 1
        self.blocks = nn.ModuleList()

        for out_ch in channels:
            # depthwise
            dw = nn.LazyConv2d(
                out_channels = prev_ch,
                kernel_size   = 3,
                stride        = 2,
                padding       = 1,
                groups        = prev_ch,
                bias          = True
            )
            # pointwise
            pw = nn.LazyConv2d(
                out_channels = out_ch,
                kernel_size   = 1,
                stride        = 1,
                padding       = 0,
                bias          = False
            )
            self.blocks.append(nn.Sequential(
                dw,
                nn.ReLU(inplace=True),
                pw,
                nn.LazyBatchNorm2d(),
                nn.ReLU(inplace=True),
            ))
            prev_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.LazyLinear(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1)


def train():
    model_name = "v0"
    batch_size = 32
    epochs = 100

    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model = TinyMotionNet()

    summary(model, (batch_size, 1, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))

    optimizer = Adam(model.parameters(), lr=0.002)

    transform = transforms.Compose([
        transforms.Resize(INPUT_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


    train_set, val_set = get_split_train_val(0.2, DATA_ROOT, "motion_other.csv", transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

    for epoch in range(epochs):
        model.train()

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, y) in train_bar:
            optimizer.zero_grad()
            y_hat = model(x.cuda())
            loss = F.mse_loss(y.cuda(), y_hat)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(mse=float(loss.item()), epoch=epoch)

        model.eval()
        val_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        loss = None
        with torch.no_grad():
            for i, (x, y) in val_bar:
                start = time.time()
                y_hat = model(x.cuda())
                speed_per_sample = (time.time() - start) / x.shape[0] * 1000
                loss = F.mse_loss(y.cuda(), y_hat)
                val_bar.set_postfix(val_mse=float(loss.item()), epoch=epoch, speed=f"{speed_per_sample:.3f}ms")

        model_save_path = f"{model_path}/{loss:.3f}.pth"
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    train()