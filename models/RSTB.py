import torch.nn as nn
from models.SwinLSTM_B import SwinTransformerBlock, PatchEmbed


class RSTB(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim, patch_size=4,
                 depth=4, num_heads=4, window_size=8,
                 mlp_ratio=4.0, drop=0.1, attn_drop=0.0, drop_path=0.0):
        super(RSTB, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        self.input_resolution = (
            img_size // patch_size, img_size // patch_size)

        self.residual_group = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path
            ) for i in range(depth)
        ])

        self.linear_proj = nn.Linear(embed_dim, embed_dim)
        self.unembed = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, in_chans,
                               kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, C, H, W]
        """
        res = x
        x = self.patch_embed(x)  # [B, N, C]

        for block in self.residual_group:
            x = block(x)

        x = self.linear_proj(x)

        # Reshape back to image space
        B, N, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.unembed(x)  # [B, in_chans, H*patch, W*patch]

        return x + res