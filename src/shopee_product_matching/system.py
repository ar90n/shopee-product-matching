from dataclasses import asdict, dataclass
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
from .neighbor import KnnMatch
from .util import save_submission_csv, get_matches


class ImageMetricLearning(pl.LightningModule):
    @dataclass
    class Param:
        start_lr: float = 1e-5
        max_lr: float = 1e-4

    def __init__(
        self,
        backbone,
        param=Param(),
        head=nn.Identity(),
        metric=nn.Identity(),
        loss=nn.CrossEntropyLoss(),
        match=KnnMatch(),
        submission_filename=None,
    ) -> None:
        super().__init__()
        self.scheduler_params = {}

        self.param = param
        self.backbone = backbone
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = head
        self.metric = metric
        self.loss = loss
        self.match = match
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
        x = batch[ShopeeProp.image]
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

            expect_matches = get_matches(
                acc_outputs["posting_ids"], acc_outputs["label_groups"]
            )
            valid_f1 = f1_score(infer_matches, expect_matches)
        except ValueError:
            valid_f1 = float("nan")

        try:
            inter_intra_class_loss = _calc_inter_intra_class_loss(
                acc_outputs["label_groups"], acc_outputs["embeddings"]
            )
        except ValueError:
            inter_intra_class_loss = float("nan")

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

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        posting_id = batch[ShopeeProp.posting_id]
        x = batch[ShopeeProp.image]
        x.to(self.device)
        y = self.forward_with_tta(x)
        return {"posting_id": posting_id, "embeddings": y}

    def test_epoch_end(self, outputs: Dict[str, List[Any]]) -> None:
        acc_outputs = _accumulate_outputs(outputs)
        index_matches = self.match(acc_outputs["embeddings"])
        matches = [
            [acc_outputs["posting_ids"][i] for i in match] for match in index_matches
        ]

        save_submission_csv(
            acc_outputs["posting_ids"], matches, self.submission_filename
        )


class TitleMetricLearning(pl.LightningModule):
    @dataclass
    class Param:
        start_lr: float = 1e-5
        max_lr: float = 1e-4

    def __init__(
        self,
        param,
        head,
        metric=nn.Identity(),
        loss=nn.CrossEntropyLoss(),
        match=KnnMatch(threshold=0.6),
        submission_filename=None,
    ) -> None:
        super().__init__()
        self.scheduler_params = {}

        self.param = param
        self.head = head
        self.metric = metric
        self.loss = loss
        self.match = match
        self.submission_filename = submission_filename

    def forward(self, x):
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch[ShopeeProp.title]
        x = x.to(self.device)
        y = batch[ShopeeProp.label_group]
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
        try:
            acc_outputs = _accumulate_outputs(outputs)
            infer_matches = find_matches(
                acc_outputs["posting_ids"], acc_outputs["embeddings"], 0.6
            )

            expect_matches = get_matches(
                acc_outputs["posting_ids"], acc_outputs["label_groups"]
            )

            valid_f1 = f1_score(infer_matches, expect_matches)
        except ValueError:
            valid_f1 = float("nan")

        self.log(
            "valid_f1",
            valid_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        embeddings = self(batch[ShopeeProp.title])
        posting_id = batch[ShopeeProp.posting_id]
        return {"posting_id": posting_id, "embeddings": embeddings}

    def test_epoch_end(self, outputs: Dict[str, List[Any]]) -> None:
        acc_outputs = _accumulate_outputs(outputs)
        matches = find_matches(
            acc_outputs["posting_ids"], acc_outputs["embeddings"], 0.6
        )

        save_submission_csv(
            acc_outputs["posting_ids"], matches, self.submission_filename
        )


def _accumulate_outputs(outputs: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
    embeddings = []
    posting_ids = []
    label_groups = []

    for output in outputs:
        embeddings.append(
            [t.detach().cpu().numpy() for t in output.get("embeddings", [])]
        )
        posting_ids.append(output.get("posting_id", []))
        label_groups.append(
            [t.detach().cpu().numpy() for t in output.get("label_group", [])]
        )
    return {
        "embeddings": np.concatenate(embeddings),
        "posting_ids": sum(posting_ids, []),
        "label_groups": np.concatenate(label_groups),
    }


def _calc_inter_intra_class_loss(
    label_groups: np.ndarray, embeddings: np.ndarray
) -> float:
    classes = sorted(np.unique(label_groups))

    centres = []
    for c in classes:
        centres.append(np.average(embeddings[label_groups == c], axis=0))

    intra_class_losses = []
    for c, centre in zip(classes, centres):
        diff = embeddings[label_groups == c] - centre
        intra_class_losses.append(np.std(diff))

    return sum(intra_class_losses) / len(classes)
