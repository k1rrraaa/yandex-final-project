import torch
import torch.nn as nn

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
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class MiniConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=16,
                 depths=None, dims=None):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2]
        if dims is None:
            dims = [64, 128, 256]
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        for i in range(2):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList([
            nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])
            for dim, depth in zip(dims, depths)
        ])

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes else nn.Identity()

    def forward_features(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
