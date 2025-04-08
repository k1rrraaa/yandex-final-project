import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # B, C, H, W


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)  # исправили на LayerNorm2d
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)  # применяем LayerNorm2d
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class MiniConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=16, depths=[2, 2, 2], dims=[64, 128, 256]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)  # исправили на LayerNorm2d
        )
        self.downsample_layers.append(stem)

        for i in range(2):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),  # исправили на LayerNorm2d
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        return self.head(x)