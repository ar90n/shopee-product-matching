import pytorch_lightning as pl
from torch import nn
import torch
from dataclasses import dataclass, asdict
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
    ):
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
        x = batch[ShopeeProp.image]
        y = batch[ShopeeProp.label_group]
        x = x.to(self.device)
        y = y.to(self.device)

        feature = self(x)
        y_hat = self.metric(feature, y)
        loss = self.loss(y_hat, y)
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss