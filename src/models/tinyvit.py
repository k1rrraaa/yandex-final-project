import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor < keep_prob
        return x.div(keep_prob) * random_tensor

class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, groups=1, bn_weight_init=1.0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self[1].weight.data.fill_(bn_weight_init)
        self[1].bias.data.zero_()

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, activation=nn.GELU):
        super().__init__()
        mid_chans = embed_dim // 2
        self.conv1 = ConvBN(in_chans, mid_chans, ks=3, stride=2, pad=1)
        self.act = activation()
        self.conv2 = ConvBN(mid_chans, embed_dim, ks=3, stride=2, pad=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio=4.0, drop_path=0.0, activation=nn.GELU):
        super().__init__()
        hidden_chans = int(in_chans * expand_ratio)
        self.conv_expand = ConvBN(in_chans, hidden_chans, ks=1)
        self.act1 = activation()
        self.conv_depthwise = ConvBN(hidden_chans, hidden_chans, ks=3, stride=1, pad=1, groups=hidden_chans)
        self.act2 = activation()
        self.conv_project = ConvBN(hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.out_chans = out_chans
        self.use_residual = (in_chans == out_chans)
        self.act3 = activation()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv_expand(x)
        out = self.act1(out)
        out = self.conv_depthwise(out)
        out = self.act2(out)
        out = self.conv_project(out)
        out = self.drop_path(out)
        if self.use_residual:
            out = identity + out
        out = self.act3(out)
        return out

class ConvStage(nn.Module):
    def __init__(self, channels, depth, drop_path_rates=None, activation=nn.GELU):
        super().__init__()
        layers = []
        drop_path_rates = drop_path_rates or [0.0] * depth
        for i in range(depth):
            layers.append(MBConv(channels, channels, expand_ratio=4.0,
                                  drop_path=drop_path_rates[i] if i < len(drop_path_rates) else 0.0,
                                  activation=activation))
        self.blocks = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

class PatchMerging(nn.Module):
    def __init__(self, in_chans, out_chans, activation=nn.GELU):
        super().__init__()
        self.conv1 = ConvBN(in_chans, out_chans, ks=1)
        self.act1 = activation()
        self.conv2 = ConvBN(out_chans, out_chans, ks=3, stride=2, pad=1, groups=out_chans)
        self.act2 = activation()
        self.conv3 = ConvBN(out_chans, out_chans, ks=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size
        relative_size = 2 * window_size - 1
        num_rel_positions = relative_size * relative_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_rel_positions, num_heads))
        coords = torch.arange(window_size)
        coords_flatten = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), dim=-1).reshape(-1, 2)
        rel_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        rel_coords += window_size - 1
        rel_index = rel_coords[:, :, 0] * (2 * window_size - 1) + rel_coords[:, :, 1]
        self.register_buffer("relative_position_index", rel_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class TinyViTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0, drop=0.0, drop_path=0.0, local_conv_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn = Attention(dim, num_heads, window_size)
        self.local_conv = ConvBN(dim, dim, ks=local_conv_size, stride=1, pad=local_conv_size//2, groups=dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, C, H_in, W_in = x.shape
        assert H_in == H and W_in == W, "Некорректный размер тензора x"
        seq = x.view(B, C, H*W).transpose(1, 2)
        shortcut = seq
        if H <= self.window_size and W <= self.window_size:
            out = self.attn(seq)
        else:
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            H_pad, W_pad = H + pad_b, W + pad_r
            x_pad = x if pad_b == 0 and pad_r == 0 else nn.functional.pad(x, (0, pad_r, 0, pad_b))
            nH = H_pad // self.window_size
            nW = W_pad // self.window_size
            x_windows = x_pad.view(B, C, nH, self.window_size, nW, self.window_size)
            x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size*self.window_size, C)
            out_windows = self.attn(x_windows)
            out_windows = out_windows.view(B, nH, nW, self.window_size, self.window_size, C)
            out_windows = out_windows.permute(0, 5, 1, 3, 2, 4)
            out_full = out_windows.reshape(B, C, H_pad, W_pad)
            if pad_b > 0 or pad_r > 0:
                out_full = out_full[:, :, :H, :W]
            out = out_full.view(B, C, H*W).transpose(1, 2)
        seq = shortcut + self.drop_path(out)
        x_local = seq.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        x_local = self.local_conv(x_local)
        seq2 = x_local.view(B, C, H*W).transpose(1, 2)
        seq = seq2 + self.drop_path(self.mlp(seq2))
        out_img = seq.transpose(1, 2).view(B, C, H, W)
        return out_img

class TinyVitStage(nn.Module):
    def __init__(self, in_chans, out_chans, depth, num_heads, window_size, mlp_ratio=4.0,
                 drop=0.0, drop_path_rates=None, downsample=None):
        super().__init__()
        self.downsample = downsample(in_chans, out_chans) if downsample is not None else nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=out_chans, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path_rates[i] if i < len(drop_path_rates) else 0.0)
            for i in range(depth)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        B, C, H, W = x.shape
        for block in self.blocks:
            x = block(x, H, W)
        return x

class TinyViT(nn.Module):
    """
    Аргументы:
        in_chans (int): Число входных каналов
        num_classes (int): Число классов для классификации
        model_size (str): Размер модели: '5M' или '11M'.
        drop_rate (float): Вероятность dropout (необязательно, по умолчанию 0).
        drop_path_rate (float): Максимальная вероятность DropPath (stochastic depth) для последних слоёв (деф. 0.1).
    """
    def __init__(self, in_chans=3, num_classes=1000, model_size='5M',
                 drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        model_size = model_size.upper()
        if model_size == '5M':
            embed_dims = [64, 128, 160, 320]
            depths = [2, 2, 6, 2]
            num_heads = [2, 4, 5, 10]
        elif model_size == '11M':
            embed_dims = [64, 128, 256, 448]
            depths = [2, 2, 6, 2]
            num_heads = [2, 4, 8, 14]
        else:
            raise ValueError("model_size должен быть '5M' или '11M'")
        window_sizes = [7, 7, 14, 7]
        mlp_ratio = 4.0

        self.patch_embed = PatchEmbed(in_chans, embed_dim=embed_dims[0])
        total_blocks = sum(depths)
        drop_path_rates = list(torch.linspace(0, drop_path_rate, total_blocks).numpy())
        dp_index = 0

        self.stages = nn.ModuleList()
        self.stages.append(ConvStage(embed_dims[0], depth=depths[0],
                                     drop_path_rates=drop_path_rates[dp_index:dp_index+depths[0]]))
        dp_index += depths[0]
        for stage_idx in range(1, len(depths)):
            in_dim = embed_dims[stage_idx - 1]
            out_dim = embed_dims[stage_idx]
            heads = num_heads[stage_idx]
            win_size = window_sizes[stage_idx]
            downsample = PatchMerging if in_dim != out_dim or stage_idx == 1 else None
            self.stages.append(TinyVitStage(in_chans=in_dim, out_chans=out_dim,
                                            depth=depths[stage_idx], num_heads=heads,
                                            window_size=win_size, mlp_ratio=mlp_ratio,
                                            drop=drop_rate,
                                            drop_path_rates=drop_path_rates[dp_index:dp_index+depths[stage_idx]],
                                            downsample=downsample))
            dp_index += depths[stage_idx]

        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).mean(dim=2)
        x = self.norm_head(x)
        x = self.head(x)
        return x
