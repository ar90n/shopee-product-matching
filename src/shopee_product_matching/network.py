import torch.nn as nn
from torch import Tensor


class AffineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activate = nn.Hardswish()

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x) -> Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        #x = self.bn(x)
        x = self.activate(x)
        return x
