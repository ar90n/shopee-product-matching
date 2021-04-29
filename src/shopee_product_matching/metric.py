import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50) -> None:
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None) -> torch.Tensor:
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.training:
            with torch.no_grad():
                B_avg = torch.where(
                    one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits)
                )
                B_avg = torch.sum(B_avg) / input.size(0)
                # print(B_avg)
                theta_med = torch.median(theta[one_hot == 1])
                self.s = torch.log(B_avg) / torch.cos(
                    torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med)
                )

        output = self.s * logits

        return output


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


def f1_score(infer_matches: List[List[str]], expect_matches: List[List[str]]) -> float:
    intersection = [
        (2 * len(set(a) & set(b))) / (len(a) + len(b))
        for a, b in zip(infer_matches, expect_matches)
    ]
    return sum(intersection) / len(intersection)
