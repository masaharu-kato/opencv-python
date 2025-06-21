import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Squeeze-and-Excitation Block (SE Block) ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Channel Attention Module (CAM) for CBAM ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # グローバル平均プーリング
        self.max_pool = nn.AdaptiveMaxPool2d(1) # グローバル最大プーリング

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)

# --- Spatial Attention Module (SAM) for CBAM ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1 
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # チャネル方向の平均プーリング
        max_out, _ = torch.max(x, dim=1, keepdim=True) # チャネル方向の最大プーリング
        x_concat = torch.cat([avg_out, max_out], dim=1) # 2チャネルに結合
        x_out = self.conv1(x_concat)
        return torch.sigmoid(x_out)

# --- CBAM (Convolutional Block Attention Module) ---
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. Channel Attention
        # CBAM論文では、元の入力xをチャンネルアテンションの結果でスケーリング
        x = self.channel_attention(x) * x 
        # 2. Spatial Attention
        # その結果を空間アテンションの結果でスケーリング
        x = self.spatial_attention(x) * x 
        return x