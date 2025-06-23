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
    
# --- ECANet (Efficient Channel Attention Network) の追加 ---
# 論文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2020)
class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 論文の推奨に基づいて、チャネル数Cに応じてカーネルサイズkを適応的に決定
        # k = | (log2(C) / gamma) + b | の最も近い奇数
        self.k = int(abs(torch.log2(torch.tensor(channel)) / gamma + b))
        if self.k % 2 == 0: # カーネルサイズは奇数にする
            self.k += 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=(self.k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x) # (B, C, 1, 1)
        # 1D畳み込みのためにテンソルの形状を変更 (B, C, 1)
        y = y.squeeze(-1).permute(0, 2, 1) # (B, 1, C)
        y = self.conv(y) # (B, 1, C)
        y = y.permute(0, 2, 1).unsqueeze(-1) # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# --- Coordinate Attention (CA) の追加 ---
# 論文: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
class Hsigmoid(nn.Module):
    """
    H-Swishの代わりに使われることが多い活性化関数
    """
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class CABlock(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CABlock, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # 高さ方向に平均プーリング
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # 幅方向に平均プーリング

        mid_channels = inp // reduction
        self.conv1 = nn.Conv2d(inp, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU() # Original paper uses Swish, but ReLU is common for simplicity

        self.conv_h = nn.Conv2d(mid_channels, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, oup, kernel_size=1, stride=1, padding=0)

        # Hsigmoid は論文で使われているが、通常のSigmoidも代替として使われる
        # 今回はHsigmoidを実装
        self.hsigmoid = Hsigmoid() # nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x) # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (N, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2) # (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.hsigmoid(self.conv_h(x_h)) # (N, C, H, 1)
        a_w = self.hsigmoid(self.conv_w(x_w)) # (N, C, 1, W)

        return identity * a_h.expand_as(identity) * a_w.expand_as(identity)
