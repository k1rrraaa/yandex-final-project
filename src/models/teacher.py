import torch
import torch.nn as nn
from timm import create_model


class ConvNeXtTeacher(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.backbone = create_model(
            'convnext_large_in22k',
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.2
        )
        self.head = nn.Sequential(
            nn.LayerNorm(1536, eps=1e-6),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.float()
        return self.head(x)

