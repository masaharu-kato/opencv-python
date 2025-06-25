import logging
from typing import Any, Literal, cast
from dataclasses import dataclass
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import random

from models.attention import CABlock, ECABlock, SEBlock, CBAM
from utils.option_utils import make_dataclass_from_cp_args

AttentionMethods = Literal[
    'Identity', # nn.Identity
    'SE_BLOCK', # Squeeze-and-Excitation Block
    'CBAM',     # Convolutional Block Attention Module
    'ECANet',   # Efficient Channel Attention Network
    'CA'        # Coordinate Attention
]

ModelOutputRange = Literal['0_to_1', 'minus1_to_1']

@dataclass
class ModelOptions:
    input_width: int
    input_height: int
    features: list[int]
    attention_method: AttentionMethods
    model_output_range: ModelOutputRange = '0_to_1'

    def input_size(self):
        return (self.input_height, self.input_width)


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
    def __init__(self, opts: ModelOptions):
        super(UNet, self).__init__()
        self.opts = opts
        
        if opts.attention_method not in AttentionMethods.__args__:
            raise RuntimeError(f'attention_method {opts.attention_method} is not supported.')

        self.input_channels = 3
        self.output_channels = 3

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_in_channels = self.input_channels
        for feature in self.opts.features:
            self.downs.append(ResBlock(current_in_channels, feature, self.opts.attention_method))
            current_in_channels = feature

        self.bottleneck = ResBlock(self.opts.features[-1], self.opts.features[-1] * 2, self.opts.attention_method)

        for i in range(len(self.opts.features) - 1, -1, -1):
            feature = self.opts.features[i]
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'), # 最近傍補間
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            self.ups.append(ResBlock(feature * 2, feature, self.opts.attention_method)) 

        self.final_conv = nn.Sequential(
            # padding_mode='reflect' is not needed here as the final conv is 1x1
            nn.Conv2d(self.opts.features[0], self.output_channels, kernel_size=1),
            nn.Sigmoid() 
        )
        opts.model_output_range = "0_to_1"


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
            
            x = upsample_block(x)

            skip_connection = skip_connections[i]

            if x.shape != skip_connection.shape:
                _, _, H_x, W_x = x.shape
                _, _, H_skip, W_skip = skip_connection.shape
                
                # スキップコネクションの方が大きい場合はクロップ
                if H_skip > H_x or W_skip > W_x:
                    diff_H = H_skip - H_x
                    diff_W = W_skip - W_x
                    skip_connection = skip_connection[:, :, diff_H // 2 : H_skip - diff_H // 2, diff_W // 2 : W_skip - diff_W // 2]
                # アップサンプリングされた方が大きい場合はクロップ (通常は起こらないはずだが念のため)
                elif H_x > H_skip or W_x > W_skip:
                    diff_H = H_x - H_skip
                    diff_W = W_x - W_skip
                    x = x[:, :, diff_H // 2 : H_x - diff_H // 2, diff_W // 2 : W_x - diff_W // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = res_block(concat_skip) 

        return self.final_conv(x)

    def save(self, path: torch.types.FileLike, **args):
        torch.save({
            'model_state_dict': self.state_dict(),
            'random_state': {
                'torch_rng_state': torch.get_rng_state(),
                'torch_cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            },
            **vars(self.opts),
            **args,
        }, path)

    @classmethod
    def load(cls, path: torch.types.FileLike, device: torch.device, args: dict[str, Any]):
        cp = torch.load(path, map_location=device, weights_only=False)
        if not (isinstance(cp, dict) and 'model_state_dict' in cp):
            raise RuntimeError("Unsupported model file.")
        
        model = cls(make_dataclass_from_cp_args(ModelOptions, cp, args)).to(device)
        model.load_state_dict(cp['model_state_dict'])

        if 'random_state' in cp:
            random_state = cp['random_state']
            torch.set_rng_state(cast(torch.Tensor, random_state['torch_rng_state']).to("cpu"))
            if random_state['torch_cuda_rng_state'] is not None:
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(cast(torch.Tensor, random_state['torch_cuda_rng_state']).to("cpu"))
                else:
                    logging.warning("Warning: CUDA RNG state is not set because CUDA is not available.")
            np.random.set_state(random_state['numpy_rng_state'])
            random.setstate(random_state['python_rng_state'])
            logging.info("Random states restored from checkpoint.")
        else:
            logging.warning("No random state found in checkpoint, using current random states.")

        logging.info(f"Loaded model: {path}")
        return model, cp
