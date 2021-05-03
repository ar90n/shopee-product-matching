# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from torchvision.transforms import (
    RandomHorizontalFlip,
    Compose,
    RandomVerticalFlip,
    RandomRotation,
    RandomApply,
    ColorJitter,
    Normalize,
    Resize,
)

from shopee_product_matching.network import AffineHead
from shopee_product_matching import constants
from shopee_product_matching.transform import (
    read_resize_normalize,
    imread,
    TitleTransform,
    label_group_encoding,
    identity,
)
from shopee_product_matching.constants import Paths
from shopee_product_matching.feature import LaserEmbedding
from shopee_product_matching.metric import create_metric
from shopee_product_matching.system import TitleMetricLearning
from shopee_product_matching.trainer import ShopeeTrainer
from shopee_product_matching.util import (
    JobType,
    get_device,
    context,
    string_escape,
)
from shopee_product_matching.datamodule import (
    ShopeeDataModule,
    ShopeeDataModuleQueries,
    ShopeeQuery,
)


# %%
def get_config_defaults() -> Dict[str, Any]:
    return {
        "train_batch_size": 16,
        "valid_batch_size": 16,
        "num_workers": 4,
        "max_epochs": 4,
        "metric": "arcface",
        "num_fc": 1,
        "num_features": 512,
        "overfit_batches": 0.01,
        "fast_dev_run": False,
        "early_stop_patience": 5,
        "fold": 0,
    }


# %%
def get_title_transforms(
    config: Any,
) -> Tuple[TitleTransform, TitleTransform, TitleTransform]:
    ind_laser_dir = Paths.requirements / "LASER"
    ind_laser_model = LaserEmbedding("id", laser_dir=ind_laser_dir)

    transform: TitleTransform = lambda x: torch.from_numpy(
        ind_laser_model.transform(string_escape(x))
    ).squeeze()
    return (transform, transform, transform)


# %%
def create_datamodule(config: Any) -> ShopeeDataModule:
    train_transform, valid_transform, test_transform = get_title_transforms(config)
    label_group_transform = label_group_encoding(config)
    queries = ShopeeDataModuleQueries(
        train=ShopeeQuery(
            title=train_transform,
            label_group=label_group_transform,
            posting_id=identity,
        ),
        valid=ShopeeQuery(
            title=valid_transform,
            label_group=label_group_transform,
            posting_id=identity,
        ),
        test=ShopeeQuery(title=test_transform),
    )

    def test_valid_split(
        df: pd.DataFrame, config: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from shopee_product_matching.constants import Paths

        fold_df = pd.read_csv(Paths.requirements / "fold.csv", index_col=0)
        train_df = df[fold_df["fold"] != config.fold]
        valid_df = df[fold_df["fold"] == config.fold]
        return train_df, valid_df

    dm = ShopeeDataModule(config, queries, test_valid_split)
    dm.setup()
    return dm


# %%
def create_system(config: Any) -> pl.LightningModule:
    metric = create_metric(
        config.metric,
        num_features=1024,
        num_classes=constants.TrainData.label_group_unique_unique_count,
    )
    head_layers = [AffineHead(1024, out_dim=config.num_features)]
    for _ in range(config.num_fc - 1):
        head_layers.append(AffineHead(config.num_features, out_dim=config.num_features))
    head = nn.Sequential(*head_layers)

    param = TitleMetricLearning.Param(max_lr=1e-5 * config.train_batch_size)
    shopee_net = TitleMetricLearning(param=param, head=head, metric=metric)
    shopee_net.to(get_device())

    return shopee_net


# %%
def create_trainer(config: Any) -> ShopeeTrainer:
    trainer = ShopeeTrainer(config, ckpt_filename_base="exp-010")
    return trainer


# %%
def train() -> None:
    config_defaults = get_config_defaults()
    with context(config_defaults, JobType.Inferene) as config:
        dm = create_datamodule(config)
        system = create_system(config)
        trainer = create_trainer(config)

        trainer.fit(system, datamodule=dm)
        trainer.save_best_model()


if __name__ == "__main__":
    #from shopee_product_matching import agent

    #agent.run(train)
    train()
