"""
Original author: Google Research.
Reimplemented from scratch for the purpose of competition participation.
This implementation does not use any pre-trained weights or external data.
"""

import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        reduced_ch = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, reduced_ch, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, stride, se_ratio=0.25, fused=False):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        mid_ch = int(in_ch * expand_ratio)
        layers = []

        if fused:
            if expand_ratio != 1:
                layers.append(ConvBNAct(in_ch, mid_ch, 3, stride))
                layers.append(ConvBNAct(mid_ch, out_ch, 1, 1, act=False))
            else:
                layers.append(ConvBNAct(in_ch, out_ch, 3, stride))
        else:
            if expand_ratio != 1:
                layers.append(ConvBNAct(in_ch, mid_ch, 1, 1))
            layers.append(ConvBNAct(mid_ch, mid_ch, 3, stride, groups=mid_ch))
            layers.append(SqueezeExcite(mid_ch, se_ratio))
            layers.append(ConvBNAct(mid_ch, out_ch, 1, 1, act=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            return x + out
        return out


class EfficientNetV2(nn.Module):
    def __init__(self, variant='M', num_classes=1000):
        super().__init__()
        config_map = {
            'M': [
                [1, 24, 3, 1, True],
                [4, 48, 5, 2, True],
                [4, 80, 5, 2, True],
                [4, 160, 7, 2, False],
                [6, 176, 14, 1, False],
                [6, 304, 18, 2, False],
            ],
            'L': [
                [1, 32, 4, 1, True],
                [4, 64, 7, 2, True],
                [4, 96, 7, 2, True],
                [4, 192, 10, 2, False],
                [6, 224, 19, 1, False],
                [6, 384, 25, 2, False],
            ],
        }
        config = config_map[variant.upper()]
        stem_ch = 24 if variant.upper() == 'M' else 32
        self.stem = ConvBNAct(3, stem_ch, 3, 2)

        blocks = []
        in_ch = stem_ch
        for expand, out_ch, n, stride, fused in config:
            stage_in_ch = in_ch
            for i in range(n):
                s = stride if i == 0 else 1
                blocks.append(MBConv(stage_in_ch if i == 0 else out_ch, out_ch, expand, s, fused=fused))
            in_ch = out_ch

        head_ch = 1280
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            ConvBNAct(in_ch, head_ch, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(head_ch, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
