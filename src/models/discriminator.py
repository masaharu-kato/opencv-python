import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int, features: list[int]):
        super().__init__()
        layers = []
        # Downsampling blocks
        # Initial Conv layer
        layers.append(nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(len(features) - 1):
            layers.append(nn.Conv2d(features[i], features[i+1], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Last Conv layer (no stride, 1x1 output for patchGAN or final prediction)
        # This layer will output 1 channel representing "realness"
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1)) # PatchGAN for 1x1 output

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
