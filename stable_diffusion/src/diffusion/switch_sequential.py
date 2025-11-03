import torch
import torch.nn as nn

import sys
sys.path.append('../')
from attention import UNetAttentionBlock
from diffusion.unet_residual_block import UNetResidualBlock


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        
        return x