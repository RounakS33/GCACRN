import torch
import torch.nn as nn


class DASG(nn.Module):
    def __init__(self, channels, reduction=2):
        super(DASG, self).__init__()
        inter_channels = channels // reduction
        self.W_g = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, inter_channels)
        )
        # Generate Attention Mask from the combined (added) features
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.GroupNorm(4, channels),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_dec, x_enc):
        # Project both to lower dimension
        g1 = self.W_g(x_dec)
        x1 = self.W_x(x_enc)
        psi = self.relu(g1 + x1)
        # Generate Mask
        mask = self.psi(psi)
        # Filter the Encoder features
        return x_dec + (x_enc * mask)
