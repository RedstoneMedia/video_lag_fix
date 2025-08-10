import os.path
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

from torchinfo import summary

from collect import DATA_ROOT
from dataset import INPUT_IMAGE_SIZE, MotionDataset
from img_process_server_connect import ImageProcessServerConnect

class TinyMotionNet(nn.Module):

    def __init__(self, y_mean=0.0, y_std=1.0, denorm=True):
        super().__init__()
        channels = [16, 40, 72, 136] # channels = [16, 40, 72, 128]
        prev_ch  = 1
        self.blocks = nn.ModuleList()
        self.denorm = denorm

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
        self.fc = nn.LazyLinear(2)

        self.register_buffer('y_mean', torch.tensor(y_mean))
        self.register_buffer('y_std', torch.tensor(y_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        mu = x[:, 0]
        log_var = x[:, 1]
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        if self.denorm:
            x = mu * self.y_std + self.y_mean
            return x.relu() # Non-negative
        return mu, log_var

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu", a=0)
                if getattr(m, "bias", None) is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, "bias", None) is not None:
                    init.zeros_(m.bias)


def train():
    model_name = "v5"
    batch_size = 32
    epochs = 250

    # Load dataset
    img_server = ImageProcessServerConnect(
        cache_dir=os.path.join(DATA_ROOT, "cache"),
        threaded_reads=True,
        working_dir=DATA_ROOT,
        fill_strategy="constant 0",
        filter_type="CatmullRom",
        grayscale=True
    )
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(
            degrees=4,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=1,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0
        )
    ])
    train_set = MotionDataset("motion_train.csv", img_server, transform)
    val_set = MotionDataset("motion_val.csv", img_server, None)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model = TinyMotionNet(y_mean=train_set.y_mean, y_std=train_set.y_std, denorm=False)
    summary(model, (batch_size, 1, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    model.initialize_weights() # after materialized run with summary

    def calc_loss(y: torch.Tensor, y_dist: tuple[torch.Tensor, torch.Tensor]):
        log_var = y_dist[1]
        log_var = torch.clamp(log_var, min=-3.0, max=1.0)
        sigma = torch.exp(0.5 * log_var) + 1e-6
        y_dist = torch.distributions.Normal(loc=y_dist[0], scale=sigma)
        return (-y_dist.log_prob(y)).mean()

    # Train
    steps = 0
    writer = SummaryWriter(log_dir=f"logs/{model_name}")
    optimizer = AdamW(model.parameters(), 3e-4, weight_decay=4e-5)
    for epoch in range(epochs):
        # Train
        model.train()
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        train_steps = 0
        train_loss_total = 0
        train_mse_total = 0
        for i, (x, y) in train_bar:
            optimizer.zero_grad()
            y_dist = model(x.cuda())
            y = y.cuda()
            loss = calc_loss(y, y_dist)
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item())
            with torch.no_grad():
                train_mse_total += float(F.mse_loss(y, y_dist[0]))

            train_bar.set_postfix(loss=float(loss.item()), epoch=epoch)
            writer.add_scalar("Loss/train", loss.item(), steps)

            steps += 1
            train_steps += 1

        writer.add_scalar("Loss/epoch-train", train_loss_total / train_steps, epoch)
        writer.add_scalar("mse/epoch-train", train_mse_total / train_steps, epoch)

        # Validation
        model.eval()
        val_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        val_loss_total = 0
        val_mse_total = 0
        val_steps = 0
        with torch.no_grad():
            for i, (x, y) in val_bar:
                start = time.time()
                y_dist = model(x.cuda())
                speed_per_sample = (time.time() - start) / x.shape[0] * 1000
                y = y.cuda()
                loss = calc_loss(y, y_dist)

                val_loss_total += float(loss.item())
                val_mse_total += float(F.mse_loss(y, y_dist[0]))

                val_bar.set_postfix(val_loss=float(loss.item()), epoch=epoch, speed=f"{speed_per_sample:.3f}ms")

                val_steps += 1

        avg_val_loss = val_loss_total / val_steps
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("mse/val", val_mse_total / val_steps, epoch)

        model_save_path = f"{model_path}/{steps}_{avg_val_loss:.3f}.pth"
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    train()