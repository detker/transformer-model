import sys
import torch
import torch.nn as nn

sys.path.append('../')
from config import CLIPConfig as Config
from clip.clip_embeddings import CLIPEmbeddings
from clip.clip_layer import CLIPLayer


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_layer = CLIPEmbeddings(Config.VOCAB_SIZE, Config.EMBD_D, Config.SEQ_LEN)
        self.layers = nn.ModuleList([CLIPLayer(Config.N_HEADS, Config.EMBD_D) for _ in range(Config.N_LAYERS)])
        self.layer_norm = nn.LayerNorm(Config.EMBD_D)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # in: (B, Seq_len) -> out: (B, Seq_len, D)
        x = x.type(torch.long)
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)

        return x
