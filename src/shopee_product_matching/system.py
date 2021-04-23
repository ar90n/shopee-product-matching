from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

from .datamodule import ShopeeProp
from .feature import find_matches
from .scheduler import ADSRScheduler
from .util import save_submission_csv


class ImageMetricLearning(pl.LightningModule):
    @dataclass
    class Param:
        start_lr: float = 1e-5
        max_lr: float = 1e-4

    def __init__(
        self,
        param,
        backbone,
        head=nn.Identity(),
        metric=nn.Identity(),
        loss=nn.CrossEntropyLoss(),
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
        self.submission_filename = submission_filename

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.head(x)
        return x

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

        x = output["embeddings"]
        x = x.to(self.device)
        y = batch[ShopeeProp.label_group]
        y = y.to(self.device)
        y_hat = self.metric(x, y)
        loss = self.loss(y_hat, y)
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {**output, "label_group": y}

    def validation_epoch_end(self, outputs: List[Dict[str, List[Any]]]) -> None:
        try:
            acc_outputs = _accumulate_outputs(outputs)
            infer_matches = find_matches(
                acc_outputs["posting_ids"], acc_outputs["embeddings"]
            )

            expect_matches = _get_expect_matches(
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
        x = batch[ShopeeProp.image]
        x.to(self.device)
        posting_id = batch[ShopeeProp.posting_id]
        return {"posting_id": posting_id, "embeddings": self(x)}

    def test_epoch_end(self, outputs: Dict[str, List[Any]]) -> None:
        acc_outputs = _accumulate_outputs(outputs)
        matches = find_matches(acc_outputs["posting_ids"], acc_outputs["embeddings"])

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


def f1_score(infer_matches: List[List[str]], expect_matches: List[List[str]]) -> float:
    intersection = [
        (2 * len(set(a) & set(b))) / (len(a) + len(b))
        for a, b in zip(infer_matches, expect_matches)
    ]
    return sum(intersection) / len(intersection)


def _get_expect_matches(
    posting_ids: List[str], label_groups: List[str]
) -> List[List[str]]:
    df = pd.DataFrame(
        {
            "label_group": label_groups,
            "posting_id": posting_ids,
        }
    )
    return (
        df["label_group"]
        .map(df.groupby(["label_group"])["posting_id"].unique().to_dict())
        .values.tolist()
    )
