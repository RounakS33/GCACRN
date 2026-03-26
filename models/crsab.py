import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module
    """

    def __init__(self, inp, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(16, inp // reduction)
        # 1x1 Conv to capture channel interactions
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.gn = nn.GroupNorm(num_groups=4, num_channels=mip)
        self.act = nn.Hardswish()
        # Split convs to project back to original channel size
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.gn(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out


class SpatialAtt(nn.Module):
    """
    Standard CBAM Spatial Attention
    """

    def __init__(self, kernel_size=7):
        super(SpatialAtt, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CRSAB(nn.Module):
    """
    Coordinate-Refined Spatial Attention Block (Novel)
    Integrates Coordinate Attention for channel/position encoding,
    followed by CBAM Spatial Attention for local refinement.
    """

    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CRSAB, self).__init__()
        # Stage 1: Coordinate Attention (Replaces CBAM Channel Attention)
        self.coord_att = CoordAtt(in_planes, reduction=reduction)
        # Stage 2: Spatial Attention (Retained from CBAM)
        self.spatial_att = SpatialAtt(kernel_size=kernel_size)

    def forward(self, x):
        # 1. Apply Coordinate Attention (Position-Aware Channel Scaling)
        x_out = self.coord_att(x)

        # 2. Apply Spatial Attention (Local Feature Refinement)
        # Note: We apply spatial attention to the output of coord attention
        scale = self.spatial_att(x_out)

        # 3. Final Output
        return x + (x_out * scale)
