from dataclasses import dataclass


@dataclass
class VAEEncoderConfig():
    IMG_CH = 3
    IN_CH = 128
    OUT_CH = 8
    SCALING_OUTPUT_CONSTANT = 0.18215