from dataclasses import dataclass 


@dataclass
class DDPMConfig():
    BETA_START = 0.00085
    BETA_END = 0.012
    TRAINING_STEPS = 1000
    INFERENCE_STEPS = 50

