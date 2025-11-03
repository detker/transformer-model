import torch
import torch.nn as nn
from torch.nn import functional as F

class TimeEmbeddings(nn.Module):
    def __init__(self, feauters):
        super().__init__()

        self.linear_1 = nn.Linear(feauters, 4*feauters)
        self.linear_2 = nn.Linear(4*feauters, 4*feauters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: (1, features=320) -> out: (1, 4*features)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x
