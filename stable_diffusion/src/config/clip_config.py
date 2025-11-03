from dataclasses import dataclass


@dataclass
class CLIPConfig():
    VOCAB_SIZE = 49408 
    EMBD_D = 768
    SEQ_LEN = 77
    N_HEADS = 12
    N_LAYERS = 12
