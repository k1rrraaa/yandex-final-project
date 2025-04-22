"""
Original author: Facebook AI Research (FAIR).
Reimplemented from scratch for the purpose of competition participation.
This implementation does not use any pre-trained weights or external data.
"""

from typing import List, Dict
import torch
import torch.nn as nn

def convnextv2_configs() -> Dict[str, Dict[str, List[int]]]:
    return {
        "tiny":  {"depths": [3, 3,  9, 3], "dims": [ 96,  192,  384,  768]},
        "small": {"depths": [3, 3, 27, 3], "dims": [ 96,  192,  384,  768]},
        "base":  {"depths": [3, 3, 27, 3], "dims": [128,  256,  512, 1024]},
    }

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(n, c, -1).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).view(n, c, h, w)
        return x

class ConvNeXtV2Block(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv   = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm     = LayerNorm2d(dim)
        self.pwconv1  = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act      = nn.GELU()
        self.pwconv2  = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma    = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        return residual + x


class ConvNeXtV2(nn.Module):
    def __init__(self, variant: str = "tiny", num_classes: int = 1000):
        super().__init__()

        if variant not in convnextv2_configs():
            raise ValueError(f"Unsupported variant '{variant}'.")
        cfg = convnextv2_configs()[variant]
        depths, dims = cfg["depths"], cfg["dims"]

        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )
        ])

        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList([
            nn.Sequential(*[ConvNeXtV2Block(dim=dims[i]) for _ in range(depths[i])])
            for i in range(4)
        ])

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.norm(x)
        return self.head(x)

def convnextv2_tiny(num_classes: int = 1000) -> ConvNeXtV2:
    return ConvNeXtV2(variant="tiny", num_classes=num_classes)


def convnextv2_small(num_classes: int = 1000) -> ConvNeXtV2:
    return ConvNeXtV2(variant="small", num_classes=num_classes)


def convnextv2_base(num_classes: int = 1000) -> ConvNeXtV2:
    return ConvNeXtV2(variant="base", num_classes=num_classes)

