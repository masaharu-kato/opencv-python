from typing import Literal, cast
import torch
import torch.nn as nn
from models.attention import CABlock, ECABlock, SEBlock, CBAM

AttentionMethods = Literal[
    'Identity', # nn.Identity
    'SE_BLOCK', # Squeeze-and-Excitation Block
    'CBAM',     # Convolutional Block Attention Module
    'ECANet',   # Efficient Channel Attention Network
    'CA'        # Coordinate Attention
]


# --- Residual Block with optional attention method ---
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, attention_method: AttentionMethods):
        super(ResBlock, self).__init__()
        
         # Set padding_mode to 'reflect' to avoid reflection padding issues
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_channels)

        if attention_method == 'CBAM':
            self.attention_module = CBAM(out_channels)
        elif attention_method == 'SE_BLOCK':
            self.attention_module = SEBlock(out_channels)
        elif attention_method == 'ECANet':
            self.attention_module = ECABlock(out_channels) # << ECABlock をインスタンス化
        elif attention_method == 'CA':
            # CA は入力と出力のチャネル数を引数にとるため、out_channels を2回渡す
            self.attention_module = CABlock(out_channels, out_channels) # << CABlock をインスタンス化
        else:
            self.attention_module = nn.Identity() # 何もしないモジュール

        # Shortcut for residual connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # 1x1 conv for shortcut doesn't typically need reflection padding
            # as it mostly handles channel dimension change.
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.final_relu = nn.ReLU(inplace=True) # Residual Add後にReLU

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention_module(out) # Apply chosen attention module
        
        out += identity # Residual Add
        out = self.final_relu(out) # Apply final ReLU
        return out

# --- UNetモデルの定義 ---
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: list[int], use_se_block=False, use_cbam=False): # << use_cbam引数を追加
        super(UNet, self).__init__()
        self.features = features
        self.use_se_block = use_se_block # 残しておくが、use_cbam=Trueなら無効化される
        self.use_cbam = use_cbam # << CBAMフラグをクラス変数に保存

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_in_channels = in_channels
        # Down part of UNet
        for feature in self.features:
            self.downs.append(ResBlock(current_in_channels, feature, self.use_se_block, self.use_cbam)) # << use_cbamを渡す
            current_in_channels = feature

        # Bottleneck
        self.bottleneck = ResBlock(self.features[-1], self.features[-1] * 2, self.use_se_block, self.use_cbam) # << use_cbamを渡す

        # Up part of UNet
        # --- 変更点: ConvTranspose2d の代わりに Upsample + Conv2d を使用 ---
        for i in range(len(self.features) - 1, -1, -1):
            feature = self.features[i]
            # 各アップサンプリングステージ用のモジュールをタプルとして追加
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'), # 最近傍補間
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            # ResBlockはスキップコネクションとアップサンプリング後の特徴マップを結合した後の処理
            self.ups.append(ResBlock(feature * 2, feature, self.use_se_block, self.use_cbam)) 
        # --- 変更ここまで ---

        self.final_conv = nn.Sequential(
            # padding_mode='reflect' is not needed here as the final conv is 1x1
            nn.Conv2d(self.opts.features[0], self.output_channels, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        skip_connections = []

        # Down path
        for down_block in self.downs:
            x = down_block(x) 
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x) 

        skip_connections = skip_connections[::-1] # Reverse skip connections

        # Up path
        for i in range(len(self.ups) // 2):
            # trans_conv は nn.Sequential で定義されたアップサンプリングブロック
            upsample_block = self.ups[i * 2] 
            res_block = self.ups[i * 2 + 1] 
            
            x = upsample_block(x) # 変更点: upsample_block を呼び出す

            skip_connection = skip_connections[i]

            # サイズが合わない場合の調整 (クロップ)
            # ConvTranspose2dを使わない場合、このクロップ処理はほとんど不要になるはずですが、
            # 念のため残しておきます。厳密なサイズ合わせはUpsampleで可能です。
            if x.shape != skip_connection.shape:
                _, _, H_x, W_x = x.shape
                _, _, H_skip, W_skip = skip_connection.shape
                
                # スキップコネクションの方が大きい場合はクロップ
                if H_skip > H_x or W_skip > W_x:
                    diff_H = H_skip - H_x
                    diff_W = W_skip - W_x
                    skip_connection = skip_connection[:, :, diff_H // 2 : H_skip - diff_H // 2,
                                                             diff_W // 2 : W_skip - diff_W // 2]
                # アップサンプリングされた方が大きい場合はクロップ (通常は起こらないはずだが念のため)
                elif H_x > H_skip or W_x > W_skip:
                    diff_H = H_x - H_skip
                    diff_W = W_x - W_skip
                    x = x[:, :, diff_H // 2 : H_x - diff_H // 2,
                                     diff_W // 2 : W_x - diff_W // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = res_block(concat_skip) 

        return self.final_conv(x)
