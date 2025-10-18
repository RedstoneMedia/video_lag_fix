import os.path
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

from torchinfo import summary

from collect import DATA_ROOT
from dataset import DuplicateDataset, INPUT_IMAGE_SIZE, make_duplicate_collate_fn
from img_process_server_connect import ImageProcessServerConnect
from models import MLPDuplicateNet, ConvDuplicateNet


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: Callable, writer, optimizer=None, metrics=None, device="cuda", total_steps=0, epoch=0):
    is_train = optimizer is not None
    phase_name = "train" if is_train else "val"
    model.train(is_train)

    total_loss = 0.0
    metric_totals = {name: 0.0 for name in (metrics or {})}
    steps = 0

    bar = tqdm(enumerate(loader), total=len(loader))
    for i, (x, y) in bar:
        x, y = x.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

        total_loss += float(loss.item())
        steps += 1

        # Compute metrics
        with torch.no_grad():
            for name, fn in (metrics or {}).items():
                metric_totals[name] += float(fn(y_hat, y).item())

        postfix = {"loss": f"{total_loss / steps:.4f}", "epoch": f"{epoch}{"" if is_train else "-val"}"}
        bar.set_postfix(**postfix)

        if is_train:
            writer.add_scalar(f"Loss/{phase_name}", loss.item(), steps + total_steps)

    avg_loss = total_loss / steps
    writer.add_scalar(f"Loss/epoch-{phase_name}", avg_loss, epoch)
    for name, total in metric_totals.items():
        writer.add_scalar(f"{name}/epoch-{phase_name}", total / steps, epoch)
    avg_metrics = {name: total / steps for name, total in metric_totals.items()}

    if is_train:
        total_steps += steps
    return total_steps, avg_loss, avg_metrics


def train():
    model_name = "conv-v4"
    batch_size = 64
    epochs = 25

    # Load dataset
    img_server = ImageProcessServerConnect(
        cache_dir=os.path.join(DATA_ROOT, "cache"),
        threaded_reads=True,
        working_dir=os.path.join(DATA_ROOT, "imgs"),
        fill_strategy="constant 0",
        filter_type="Nearest",
        grayscale=True
    )
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
    ])
    train_collate = make_duplicate_collate_fn(img_server, transform=transform)
    val_collate = make_duplicate_collate_fn(img_server, transform=None)
    train_set = DuplicateDataset(DATA_ROOT, "train.csv")
    val_set = DuplicateDataset(DATA_ROOT, "val.csv")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=val_collate, num_workers=0, pin_memory=True)

    # Create model
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model = ConvDuplicateNet(channels=(4, 8, 16, 32))
    summary(model, (batch_size, 2, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))

    # Train
    writer = SummaryWriter(log_dir=f"logs/{model_name}")
    optimizer = AdamW(model.parameters())
    loss_fn = F.binary_cross_entropy_with_logits
    metrics = {}
    steps = 0
    for epoch in range(epochs):
        steps, train_loss, train_metrics = run_epoch(model, train_loader, loss_fn, writer, optimizer, metrics, total_steps=steps, epoch=epoch)
        _, val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, writer, None, metrics,epoch=epoch)

        model_save_path = f"{model_path}/{steps}_{val_loss:.3f}.pth"
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    train()