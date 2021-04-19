from dataclasses import asdict, dataclass
from typing import List, Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
import pandas as pd

# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from torch._C import Value

from .datamodule import ShopeeProp
from .scheduler import ADSRScheduler


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
    ) -> None:
        super().__init__()
        self.scheduler_params = {}

        self.param = param
        self.backbone = backbone
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = head
        self.metric = metric
        self.loss = loss

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
            flatten_outputs = self._accumulate_outputs(outputs)
            infer_label_groups = get_image_predictions(
                flatten_outputs["posting_ids"], flatten_outputs["embeddings"]
            )

            df = pd.DataFrame(
                {
                    "label_group": flatten_outputs["label_groups"],
                    "posting_id": flatten_outputs["posting_ids"],
                }
            )
            matches = df["label_group"].map(
                df.groupby(["label_group"])["posting_id"].unique().to_dict()
            )
            intersection = np.array([(2 * len(set(a) & set(b))) / (len(a) + len(b)) for a,b in zip(matches, infer_label_groups)])
            valid_f1_score = sum(intersection) / len(intersection)
        except ValueError:
            valid_f1_score = float('nan')

        self.log(
            "valid_f1_score", valid_f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        x = batch[ShopeeProp.image]
        x.to(self.device)
        posting_id = batch[ShopeeProp.posting_id]
        return {"posting_id": posting_id, "embeddings": self(x)}

    def test_epoch_end(self, outputs: Dict[str, List[Any]]) -> None:
        flatten_outputs = self._accumulate_outputs(outputs)
        result = get_image_predictions(
            flatten_outputs["posting_ids"], flatten_outputs["embeddings"]
        )
        df = pd.DataFrame(
            {
                "posting_id": flatten_outputs["posting_ids"],
                "matches": [" ".join(rs) for rs in result]
            }
        )
        df.to_csv('submission.csv', index = False)
        print(result)

    def _accumulate_outputs(
        self, outputs: List[Dict[str, List[Any]]]
    ) -> Dict[str, Any]:
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


def get_image_predictions(posting_ids, embeddings, threshold=3.4) -> List[List[str]]:
    KNN = min(3, len(posting_ids))

    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    predictions = []
    for k in range(embeddings.shape[0]):
        idx = np.where(
            distances[
                k,
            ]
            < threshold
        )[0]
        ids = indices[k, idx]
        predictions.append([posting_ids[i] for i in ids])

    return predictions