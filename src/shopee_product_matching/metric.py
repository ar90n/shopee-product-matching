import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.scalar_tensor(math.cos(self.m)).to(torch.float16)
        self.sin_m = torch.scalar_tensor(math.sin(self.m)).to(torch.float16)
        self.th = torch.scalar_tensor(math.cos(math.pi - self.m)).to(torch.float16)
        self.mm = torch.scalar_tensor(math.sin(math.pi - self.m) * self.m).to(
            torch.float16
        )

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.to(torch.float16)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        sine = sine.to(torch.float16)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.to(torch.float16)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
