import torch
import torch.nn as nn
from models.SwinLSTM_D import SwinTransformerBlock, PatchEmbed
import collections.abc


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple([x] * 2)


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, out_chans):
        super(PatchUnEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.embed_dim = embed_dim
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.grid_size[0], self.grid_size[1])
        x = self.proj(x)
        return x


class RSTB(nn.Module):
    def __init__(self, input_resolution, in_channels, dim, depth, num_heads, window_size,
                 patch_size=1, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = (
            input_resolution[0] // patch_size, input_resolution[1] // patch_size)
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=input_resolution,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.residual_group = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=self.input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path if isinstance(
                    drop_path, float) else drop_path[i]
            ) for i in range(depth)
        ])
        self.patch_unembed = PatchUnEmbed(
            img_size=input_resolution,
            patch_size=patch_size,
            embed_dim=dim,
            out_chans=in_channels
        )

    def forward(self, x):
        shortcut = x
        x = self.patch_embed(x)
        for block in self.residual_group:
            x = block(x)
        x = self.patch_unembed(x)
        return x + shortcut
