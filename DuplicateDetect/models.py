from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def mae_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b, reduction="none").mean(dim=[1, 2, 3])

def mse_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="none").mean(dim=[1, 2, 3])

def laplacian_energy_diff_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3)
    if a.size(1) > 1:
        kernel = kernel.repeat(a.size(1), 1, 1, 1)
    def lap(x):
        return F.conv2d(x, kernel, padding=1, groups=a.size(1))
    la, lb = lap(a), lap(b)
    return F.l1_loss(la, lb, reduction="none").mean(dim=[1, 2, 3])

def fft_energy_diff_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    fa = torch.fft.rfft2(a, dim=[2, 3])
    fb = torch.fft.rfft2(b, dim=[2, 3])
    mag_a = fa.abs()
    mag_b = fb.abs()
    return F.l1_loss(mag_a, mag_b, reduction="none").mean(dim=[1, 2, 3])

def block_mae_fn(a: torch.Tensor, b: torch.Tensor, block: int = 8) -> torch.Tensor:
    N, C, H, W = a.shape
    h_blocks = H // block
    w_blocks = W // block
    a_blocks = a[:, :, :h_blocks*block, :w_blocks*block].reshape(N, C, h_blocks, block, w_blocks, block)
    b_blocks = b[:, :, :h_blocks*block, :w_blocks*block].reshape(N, C, h_blocks, block, w_blocks, block)
    diff = (a_blocks - b_blocks).abs().mean(dim=(1, 3, 5))
    return diff.view(N, -1)

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(1) @ g.unsqueeze(0)
    kernel = kernel_1d.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    return kernel

def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    window_size = 11
    sigma = 1.5
    kernel = _gaussian_kernel(window_size, sigma)
    pad = window_size // 2
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)


    k = kernel.to(device=a.device, dtype=a.dtype)
    # compute local means
    mu1 = F.conv2d(a, k, padding=pad, groups=1)
    mu2 = F.conv2d(b, k, padding=pad, groups=1)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(a * a, k, padding=pad, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(b * b, k, padding=pad, groups=1) - mu2_sq
    sigma12 = F.conv2d(a * b, k, padding=pad, groups=1) - mu1_mu2

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean(dim=[1, 2, 3])


class MLPDuplicateNet(nn.Module):

    def __init__(self, metric_fns: Optional[dict[str, Callable]] = None, hidden_sizes=(8, 8)):
        super().__init__()
        self.fc = nn.Sequential()
        for n in hidden_sizes:
            self.fc.append(nn.LazyLinear(n))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.LazyLinear(1))

        if metric_fns is None:
            metric_fns = {
                "mae": mae_fn,
                "mse": mse_fn,
                #"ssim": ssim,
                "laplacian_energy_diff": laplacian_energy_diff_fn,
                "fft_energy_diff": fft_energy_diff_fn,
                "block_mae": block_mae_fn,
            }
        self.metric_fns = metric_fns

    def compute_metrics(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        names = list(self.metric_fns.keys())
        vals = []
        with torch.no_grad():
            for n in names:
                if n not in self.metric_fns:
                    raise KeyError(f"Metric '{n}' not provided")
                v = self.metric_fns[n](a, b)
                if not torch.is_tensor(v):
                    v = torch.tensor(v, device=a.device, dtype=a.dtype)
                v = v.to(device=a.device, dtype=a.dtype)
                if v.ndim == 1:
                    v = v.view(-1, 1)
                else:
                    v = v.view(v.shape[0], -1)
                vals.append(v)
            if len(vals) == 0:
                return torch.empty((a.shape[0], 0), device=a.device, dtype=a.dtype)
            metrics = torch.cat(vals, dim=1)
        return metrics.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before = x[:, 0, ...].unsqueeze(1)
        after = x[:, 1, ...].unsqueeze(1)
        metrics = self.compute_metrics(before, after)
        return self.fc(metrics)


class ConvDuplicateNet(nn.Module):

    def __init__(self, channels=(4, 8, 16), global_pool_size: Optional[int] = None):
        super().__init__()

        prev_ch  = 1
        self.blocks = nn.ModuleList()

        for out_ch in channels:
            # pointwise
            pw = nn.LazyConv2d(
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            # depthwise
            dw = nn.LazyConv2d(
                out_channels = prev_ch,
                kernel_size   = 3,
                stride        = 2,
                padding       = 1,
                groups        = prev_ch,
                bias          = True
            )
            self.blocks.append(nn.Sequential(
                pw,
                nn.LazyBatchNorm2d(),
                nn.ReLU(inplace=True),
                dw,
                nn.ReLU(inplace=True)
            ))
            prev_ch = out_ch

        if global_pool_size:
            self.global_pool = nn.AdaptiveAvgPool2d(global_pool_size)
        else:
            self.global_pool = None
        self.fc = nn.Sequential(
            nn.LazyLinear(8),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before = x[:, 0, ...].unsqueeze(1)
        after = x[:, 1, ...].unsqueeze(1)
        x = torch.abs(before - after)

        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
