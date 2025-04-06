# This is a from-scratch reimplementation of ConvNeXt, originally described in:
# "A ConvNet for the 2020s" - Liu et al., https://arxiv.org/abs/2201.03545
# The code was written without using pretrained weights or direct copying.
# Rewritten by Kirill Samigullin, 2025, for use in a transfer-learning-restricted competition.


import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=1000, drop_path_rate=0.):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                LayerNorm(96)
            ),
            nn.Sequential(
                LayerNorm(96),
                nn.Conv2d(96, 192, kernel_size=2, stride=2)
            ),
            nn.Sequential(
                LayerNorm(192),
                nn.Conv2d(192, 384, kernel_size=2, stride=2)
            ),
            nn.Sequential(
                LayerNorm(384),
                nn.Conv2d(384, 768, kernel_size=2, stride=2)
            )
        ])

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=96 * 2**i, drop_path=dp_rates[cur + j]) for j in range([3, 3, 9, 3][i])]
            )
            self.stages.append(stage)
            cur += [3, 3, 9, 3][i]

        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)
        
        self.apply(self._init_weights)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))


def convnext_tiny(num_classes=1000):
    return ConvNeXtTiny(num_classes=num_classes)