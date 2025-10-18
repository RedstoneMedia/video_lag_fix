from typing import Optional
import torch
import torch.nn as nn


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
