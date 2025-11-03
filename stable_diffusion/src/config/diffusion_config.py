from dataclasses import dataclass


@dataclass
class DiffusionConfig():
    D_TIME = 320
    D_UNET_DECODER_OUT = 320
    D_UNET_ENC_IN = 320
    D_DIFFUSION_IN_OUT = 4
    N_HEADS = 8
    HEAD_SIZE = 40