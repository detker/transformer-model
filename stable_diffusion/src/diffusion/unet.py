import sys
import torch
import torch.nn as nn

sys.path.append('../')
from config import DiffusionConfig as Config
from attention import UNetAttentionBlock
from diffusion.unet_residual_block import UNetResidualBlock
from diffusion.upsample import Upsample
from diffusion.switch_sequential import SwitchSequential


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # (B, 4, H/8, W/8) -> (B, 1280, H/64, W/64)
        self.encoder = nn.ModuleList([
            # (B, 320, H/8, W/8)
            SwitchSequential(nn.Conv2d(Config.D_DIFFUSION_IN_OUT, Config.D_UNET_ENC_IN, kernel_size=3, padding=1)),
            
            SwitchSequential(UNetResidualBlock(Config.D_UNET_ENC_IN, Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, Config.HEAD_SIZE)),

            SwitchSequential(UNetResidualBlock(Config.D_UNET_ENC_IN, Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, Config.HEAD_SIZE)),

            # (B, 320, H/16, W/16)
            SwitchSequential(nn.Conv2d(Config.D_UNET_ENC_IN, Config.D_UNET_ENC_IN, kernel_size=3, stride=2, padding=1)),

            # (B, 640, H/16, W/16)
            SwitchSequential(UNetResidualBlock(Config.D_UNET_ENC_IN, 2*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 2*Config.HEAD_SIZE)),

            SwitchSequential(UNetResidualBlock(2*Config.D_UNET_ENC_IN, 2*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 2*Config.HEAD_SIZE)),

            # (B, 640, H/32, W/32)
            SwitchSequential(nn.Conv2d(2*Config.D_UNET_ENC_IN, 2*Config.D_UNET_ENC_IN, kernel_size=3, stride=2, padding=1)),

            # (B, 1280, H/32, W/32) 
            SwitchSequential(UNetResidualBlock(2*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 4*Config.HEAD_SIZE)),
            
            SwitchSequential(UNetResidualBlock(4*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 4*Config.HEAD_SIZE)),

            # (B, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(4*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNetResidualBlock(4*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN)),

            SwitchSequential(UNetResidualBlock(4*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN))
        ])

        # (B, 1280, H/64, W/64) -> (B, 1280, H/64, W/64)
        self.bottleneck = SwitchSequential(
            UNetResidualBlock(4*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN),
            UNetAttentionBlock(Config.N_HEADS, 4*Config.HEAD_SIZE),
            UNetResidualBlock(4*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN)
        )

        # (B, 1280, H/64, W/64) -> (B, 320, H/8, W/8)
        self.decoder = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(8*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN)),

            SwitchSequential(UNetResidualBlock(8*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN)),

            # (B, 1280, H/32, W/32)
            SwitchSequential(UNetResidualBlock(8*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN),
                             Upsample(4*Config.D_UNET_ENC_IN)),

            SwitchSequential(UNetResidualBlock(8*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 4*Config.HEAD_SIZE)),

            SwitchSequential(UNetResidualBlock(8*Config.D_UNET_ENC_IN, 4*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 4*Config.HEAD_SIZE)),

            # (B, 1280, H/16, W/16)
            SwitchSequential(UNetResidualBlock(4*Config.D_UNET_ENC_IN+2*Config.D_UNET_ENC_IN,
                                               4*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 4*Config.HEAD_SIZE),
                             Upsample(4*Config.D_UNET_ENC_IN)),

            SwitchSequential(UNetResidualBlock(4*Config.D_UNET_ENC_IN+2*Config.D_UNET_ENC_IN,
                                               2*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 2*Config.HEAD_SIZE)),

            SwitchSequential(UNetResidualBlock(4*Config.D_UNET_ENC_IN, 2*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 2*Config.HEAD_SIZE)),

            # (B, 640, H/8, W/8)                             
            SwitchSequential(UNetResidualBlock(2*Config.D_UNET_ENC_IN+Config.D_UNET_ENC_IN, 
                                               2*Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, 2*Config.HEAD_SIZE),
                             Upsample(2*Config.D_UNET_ENC_IN)),

            SwitchSequential(UNetResidualBlock(2*Config.D_UNET_ENC_IN+Config.D_UNET_ENC_IN,
                                               Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, Config.HEAD_SIZE)),

            SwitchSequential(UNetResidualBlock(2*Config.D_UNET_ENC_IN, Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, Config.HEAD_SIZE)),

            # (B, 320, H/8, W/8)                 
            SwitchSequential(UNetResidualBlock(2*Config.D_UNET_ENC_IN, Config.D_UNET_ENC_IN),
                             UNetAttentionBlock(Config.N_HEADS, Config.HEAD_SIZE))
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)        

        residual_connections = []

        for layer in self.encoder:
            x = layer(x, context, time)
            residual_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoder:
            x = torch.cat([x, residual_connections.pop()], dim=1)
            x = layer(x, context, time) 

        return x
