"""
PETALface: Swin Transformer + Dual LoRA face recognition model
Reference: https://github.com/Kartik-3004/PETALface
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class LoRALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, r: int, scale: float, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.r = r
        self.trainable_lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.trainable_lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale

        nn.init.normal_(self.trainable_lora_down.weight, std=1 / r)
        nn.init.zeros_(self.trainable_lora_up.weight)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        lora = self.scale * self.dropout(self.trainable_lora_up(self.trainable_lora_down(x)))
        return out + lora


class LoRALinearTwo(nn.Linear):
    def __init__(self, in_features: int, out_features: int, r: int, scale: float, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.r = r
        self.trainable_lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.trainable_lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale

        nn.init.normal_(self.trainable_lora_down.weight, std=1 / r)
        nn.init.zeros_(self.trainable_lora_up.weight)

        self.trainable_lora_down2 = nn.Linear(in_features, r, bias=False)
        self.dropout2 = nn.Dropout(0.1)
        self.trainable_lora_up2 = nn.Linear(r, out_features, bias=False)
        self.scale2 = scale

        nn.init.normal_(self.trainable_lora_down2.weight, std=1 / self.r)
        nn.init.zeros_(self.trainable_lora_up2.weight)

    def forward(self, x, alpha):
        out = F.linear(x, self.weight, self.bias)
        lora_1 = self.scale * self.dropout(self.trainable_lora_up(self.trainable_lora_down((1 - alpha) * x)))
        lora_2 = self.scale2 * self.dropout2(self.trainable_lora_up2(self.trainable_lora_down2(alpha * x)))
        return out + lora_1 + lora_2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, lora_rank, lora_scale, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., use_lora=False):
        super().__init__()
        self.dim = dim
        self.use_lora = use_lora
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        if self.use_lora:
            self.qkv = LoRALinearTwo(dim, dim * 3, r=lora_rank, scale=lora_scale, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, alpha, mask=None):
        B_, N, C = x.shape
        n_windows = int(x.shape[0] / alpha.shape[0])
        alpha = alpha.repeat_interleave(n_windows, dim=0)

        if self.use_lora:
            qkv = self.qkv(x, alpha.view(-1, 1, 1))
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn[:, :, :self.window_size[0] * self.window_size[1], :self.window_size[0] * self.window_size[1]] = \
            attn[:, :, :self.window_size[0] * self.window_size[1], :self.window_size[0] * self.window_size[1]] + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, lora_rank, lora_scale, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_lora=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_lora = use_lora

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            lora_rank=lora_rank, lora_scale=lora_scale,
            dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_lora=use_lora)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, alpha):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, alpha, mask=self.attn_mask)
        attn_windows = attn_windows[:, :25]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, lora_rank, lora_scale, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_lora=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                lora_rank=lora_rank, lora_scale=lora_scale,
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_lora=use_lora)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution=input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, alpha):
        for blk in self.blocks:
            x = blk(x, alpha)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, lora_rank=4, lora_scale=1.0, img_size=112, patch_size=9, in_chans=3, num_classes=512,
                 embed_dim=128, depths=(8, 12, 2), num_heads=(4, 16, 16),
                 window_size=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.05,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True, use_lora=False, reso=120):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.use_lora = use_lora
        self.reso = reso

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            res = patches_resolution[0] // (2 ** i_layer)
            layer = BasicLayer(
                lora_rank=lora_rank, lora_scale=lora_scale,
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(res, res),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_lora=self.use_lora)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        if self.use_lora:
            self.feature_layer = nn.Sequential(
                nn.BatchNorm1d(embed_dim * 100), nn.Dropout(0.25),
                LoRALinear(embed_dim * 100, 512, r=lora_rank, scale=lora_scale),
                nn.BatchNorm1d(512))
        else:
            self.feature_layer = nn.Sequential(
                nn.BatchNorm1d(embed_dim * 100), nn.Dropout(0.25),
                nn.Linear(embed_dim * 100, 512),
                nn.BatchNorm1d(512))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, alpha):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, alpha)
        x = self.norm(x)
        B, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        return x

    def forward(self, x, alpha):
        x = self.forward_features(x, alpha)
        if self.reso == 112:
            x = self.feature1(x)
            x = self.feature2(x)
        else:
            x = self.feature_layer(x)
        return x


_MODEL_REGISTRY = {}


def _build_swin_iqa(**kwargs):
    return SwinTransformer(
        lora_rank=kwargs.get("lora_rank", 8),
        lora_scale=kwargs.get("lora_scale", 1.0),
        img_size=120,
        patch_size=6,
        in_chans=3,
        num_classes=512,
        embed_dim=384,
        depths=(2, 18, 2),
        num_heads=(8, 16, 16),
        window_size=5,
        use_lora=kwargs.get("use_lora", True),
        reso=120,
    )


_MODEL_REGISTRY["swin_arcface_webface4m_tinyface"] = _build_swin_iqa
_MODEL_REGISTRY["swin_cosface_webface4m_tinyface"] = _build_swin_iqa
_MODEL_REGISTRY["swin_cosface_webface4m_briar"] = _build_swin_iqa
_MODEL_REGISTRY["swin_cosface_webface12m_briar"] = _build_swin_iqa
_MODEL_REGISTRY["swin_arcface_webface4m"] = _build_swin_iqa
_MODEL_REGISTRY["swin_cosface_webface4m"] = _build_swin_iqa
_MODEL_REGISTRY["swin_arcface_webface12m"] = _build_swin_iqa
_MODEL_REGISTRY["swin_cosface_webface12m"] = _build_swin_iqa


def create_model(model_name: str, lora_rank: int = 8, lora_scale: float = 1.0, use_lora: bool = True) -> nn.Module:
    builder = _MODEL_REGISTRY.get(model_name)
    if builder is None:
        msg = f"未知模型: {model_name}，可用: {list(_MODEL_REGISTRY.keys())}"
        raise ValueError(msg)
    return builder(lora_rank=lora_rank, lora_scale=lora_scale, use_lora=use_lora)


def load_model(model_path: str, model_name: str = "swin_arcface_webface4m_tinyface",
               lora_rank: int = 8, lora_scale: float = 1.0, use_lora: bool = True,
               device: str = "cpu") -> nn.Module:
    model = create_model(model_name, lora_rank=lora_rank, lora_scale=lora_scale, use_lora=use_lora)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] 缺失权重: {missing}")
    if unexpected:
        print(f"[WARN] 多余权重: {unexpected}")
    model.eval()
    return model
