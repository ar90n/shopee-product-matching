from dataclasses import asdict, dataclass
from functools import total_ordering
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

from .datamodule import ShopeeProp
from .feature import find_matches
from .metric import f1_score
from .scheduler import ADSRScheduler
from .neighbor import CosineSimilarityMatch, KnnMatch
from .util import (
    get_device,
    save_submission_csv,
    save_submission_confidence,
    save_submission_embedding,
    get_matches,
)


class ImageMetricLearning(pl.LightningModule):
    @dataclass
    class Param:
        start_lr: float = 1e-5
        max_lr: float = 1e-4

    def __init__(
        self,
        param=Param(),
        backbone=nn.Identity(),
        pooling=nn.AdaptiveAvgPool2d(1),
        head=nn.Identity(),
        metric=nn.Identity(),
        loss=nn.CrossEntropyLoss(),
        match=CosineSimilarityMatch(0.25),
        source_prop: ShopeeProp = ShopeeProp.image,
        save_submission_embedding: bool = False,
        save_submission_confidence: bool = False,
        submission_filename=None,
    ) -> None:
        super().__init__()
        self.scheduler_params = {}

        self.param = param
        self.backbone = backbone
        self.pooling = pooling
        self.head = head
        self.metric = metric
        self.loss = loss
        self.match = match
        self.source_prop = source_prop
        self.save_submission_embedding = save_submission_embedding
        self.save_submission_confidence = save_submission_confidence
        self.submission_filename = submission_filename

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.head(x)
        return x

    def forward_with_tta(self, x):
        dims = x.shape
        x = x.reshape([-1, *dims[-3:]])
        y = self(x)
        y = y.reshape([*dims[:-3], -1])
        if len(dims) == 5:
            y = torch.mean(y, axis=1)
        return y

    def training_step(self, batch, batch_idx):
        x = batch[self.source_prop]
        y = batch[ShopeeProp.label_group]
        x = x.to(self.device)
        y = y.to(self.device)

        feature = self(x)
        y_hat = self.metric(feature, y)
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param.start_lr)
        scheduler = ADSRScheduler(optimizer, **asdict(self.param))
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        output = self.test_step(batch, batch_idx)
        y = batch[ShopeeProp.label_group]
        y = y.to(self.device)
        return {**output, "label_group": y}

    def validation_epoch_end(self, outputs: List[Dict[str, List[Any]]]) -> None:
        acc_outputs = _accumulate_outputs(outputs)
        try:
            index_matches = self.match(acc_outputs["embeddings"])
            infer_matches = [
                [acc_outputs["posting_ids"][i] for i in match]
                for match in index_matches
            ]

            label_groups = acc_outputs["label_groups"].detach().cpu().numpy().squeeze()
            expect_matches = get_matches(
                acc_outputs["posting_ids"], label_groups
            )
            valid_f1 = f1_score(infer_matches, expect_matches)
        except ValueError:
            valid_f1 = float("nan")

        try:
            (
                inter_intra_class_loss,
                intra_loss,
                inter_loss,
                intra_max,
                inter_min,
                intra_quar,
                inter_quar,
            ) = _calc_inter_intra_class_loss(
                acc_outputs["label_groups"], acc_outputs["embeddings"]
            )
        except ValueError:
            inter_intra_class_loss = float("nan")
            intra_loss = float("nan")
            inter_loss = float("nan")
            intra_max = float("nan")
            inter_min = float("nan")
            intra_quar = float("nan")
            inter_quar = float("nan")

        self.log(
            "valid_f1",
            valid_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "valid_loss",
            inter_intra_class_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "valid_intra_loss",
            intra_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_inter_loss",
            inter_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_intra_max",
            intra_max,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_inter_min",
            inter_min,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_intra_quar",
            intra_quar,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_inter_quar",
            inter_quar,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        posting_id = batch[ShopeeProp.posting_id]
        x = batch[self.source_prop]
        x.to(self.device)
        if self.source_prop == ShopeeProp.image:
            y = self.forward_with_tta(x)
        else:
            y = self(x)
        return {"posting_id": posting_id, "embeddings": y}

    def test_epoch_end(self, outputs: Dict[str, List[Any]]) -> None:
        acc_outputs = _accumulate_outputs(outputs)

        if self.save_submission_confidence:
            save_submission_confidence(
                acc_outputs["posting_ids"],
                acc_outputs["embeddings"],
                self.match._threshold,
                self.submission_filename,
            )
        else:
            index_matches = self.match(acc_outputs["embeddings"])
            matches = [
                [acc_outputs["posting_ids"][i] for i in match] for match in index_matches
            ]

            save_submission_csv(
                acc_outputs["posting_ids"], matches, self.submission_filename
            )
        if self.save_submission_embedding:
            save_submission_embedding(
                acc_outputs["posting_ids"],
                acc_outputs["embeddings"],
                self.submission_filename,
            )


def _accumulate_outputs(outputs: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
    embeddings = []
    posting_ids = []
    label_groups = []

    for output in outputs:
        embeddings.append(
            [t for t in output.get("embeddings", [])]
        )
        posting_ids.append(output.get("posting_id", []))
        label_groups.append(
            [t for t in output.get("label_group", [])]
        )
    ret = {
        "embeddings": torch.vstack(sum(embeddings, [])),
        "posting_ids": sum(posting_ids, []),
    }
    if 0 < len(label_groups[0]):
        ret["label_groups"] =torch.vstack(sum(label_groups, [])) 
    return ret


def _calc_inter_intra_class_loss(
    label_groups: torch.Tensor, embeddings: torch.Tensor
) -> float:
    label_groups = label_groups.squeeze()
    classes = sorted(torch.unique(label_groups).detach().cpu().numpy().squeeze())

    embeddings = embeddings.to(torch.float32)
    embeddings *= 1.0 / (torch.norm(embeddings, dim=1).reshape(-1, 1) + 1e-12)

    centres = []
    for c in classes:
        centres.append(torch.mean(embeddings[label_groups == c], axis=0))
    centres = torch.stack(centres, axis=0)

    intra_losses = []
    inter_losses = []
    intra_inter_class_losses = []
    for c, centre in zip(classes, centres):
        intra_class_distances = (
            1.0 - torch.matmul(embeddings[label_groups == c], centre.T).cpu().T
        )
        mean_intra_class_dist = torch.mean(intra_class_distances)
        intra_losses.append(mean_intra_class_dist)

        inter_class_distances = 1.0 - torch.matmul(centres, centre.T).cpu().T
        mean_inter_class_dist = torch.mean(inter_class_distances)
        inter_losses.append(mean_inter_class_dist)

        loss = mean_intra_class_dist / (mean_inter_class_dist + 1e-12)
        intra_inter_class_losses.append(loss)

    n = len(classes)
    intra_loss = sum(intra_losses) / n
    inter_loss = sum(inter_losses) / n
    intra_max = max(intra_losses)
    inter_min = min(inter_losses)
    intra_quar = sorted(intra_losses)[-(n // 4)]
    inter_quar = sorted(inter_losses)[(n // 4)]
    total_loss = sum(intra_inter_class_losses) / n
    return (
        total_loss,
        intra_loss,
        inter_loss,
        intra_max,
        inter_min,
        intra_quar,
        inter_quar,
    )
