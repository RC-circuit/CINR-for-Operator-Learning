import torch.nn as nn
import torch
import numpy as np

class CNNEncoderBlock(nn.Module):
    def __init__(self, channels, latent_size, kernel_size=3, stride=1, padding=1, use_bn=True, use_pool=True):
        super().__init__()
        layers = []
        n_layers = len(channels)-1
        for n in range(n_layers):
          layers.append(nn.Conv2d(channels[n], channels[n+1], kernel_size=kernel_size, stride=stride, padding=padding))
          if use_bn:
              layers.append(nn.BatchNorm2d(channels[n+1]))
          layers.append(nn.GELU())
          if use_pool:
              layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_head = nn.Sequential(nn.Flatten(),
                                          nn.Linear(channels[-1], latent_size)
                                          )

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        out = self.encoder_head(x)
        return out