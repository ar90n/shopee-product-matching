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

import timm
import torch
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

from shopee_product_matching import constants
from shopee_product_matching.neighbor import CosineSimilarityMatch
from shopee_product_matching.network import AffineHead
from shopee_product_matching.transform import (
    read_resize_normalize,
    imread,
    ImageTransform,
    label_group_encoding,
    identity,
)
from shopee_product_matching.metric import ArcMarginProduct
from shopee_product_matching.system import ImageMetricLearning
from shopee_product_matching.trainer import ShopeeTrainer
from shopee_product_matching.util import (
    JobType,
    get_device,
    context,
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
        "max_epochs": 12,
        #"backbone": "efficientnet_v2s",
        "backbone": "eca_nfnet_l0",
        "dim_embedding": 512, 
        "image_size": 512,
        "overfit_batches": 0,
        "fast_dev_run": False,
        "early_stop_patience": 15,
        "match_threshold": 0.25,
        "fold": 0,
    }


# %%
def get_image_transforms(
    config: Any,
) -> Tuple[ImageTransform, ImageTransform, ImageTransform]:
    train_transform = Compose(
        [
            imread("train_images"),
            Resize(size=(config.image_size, config.image_size)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomApply(torch.nn.ModuleList([RandomRotation(degrees=120)]), p=0.8),
            RandomApply(
                torch.nn.ModuleList([ColorJitter(brightness=(0.09, 0.6))]), p=0.5
            ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    valid_transform = read_resize_normalize(config, "train_images")
    test_transform = read_resize_normalize(config, "test_images")
    return (train_transform, valid_transform, test_transform)


# %%
def create_datamodule(config: Any) -> ShopeeDataModule:
    train_transform, valid_transform, test_transform = get_image_transforms(config)
    label_group_transform = label_group_encoding(config)
    queries = ShopeeDataModuleQueries(
        train=ShopeeQuery(
            image=train_transform,
            label_group=label_group_transform,
            posting_id=identity,
        ),
        valid=ShopeeQuery(
            image=valid_transform,
            label_group=identity,
            posting_id=identity,
        ),
        test=ShopeeQuery(image=test_transform),
    )

    def test_valid_split(
        df: pd.DataFrame, config: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from shopee_product_matching.constants import Paths

        fold_df = pd.read_csv(Paths.requirements / "fold.csv", index_col=0)
        train_df = df[fold_df["fold"] != config.fold]
        valid_df = df[fold_df["fold"] == config.fold]
        return train_df, valid_df

    return ShopeeDataModule(config, queries, test_valid_split)


# %%
def create_system(config: Any) -> pl.LightningModule:
    backbone = timm.create_model(
        config.backbone, pretrained=True, num_classes=0, global_pool=""
    )
    head = AffineHead(backbone.num_features, config.dim_embedding)
    metric = ArcMarginProduct(
        config.dim_embedding,
        constants.TrainData.label_group_unique_count_pdf_fold,
    )
    match = CosineSimilarityMatch(config.match_threshold)
    param = ImageMetricLearning.Param(max_lr=1e-5 * config.train_batch_size)
    shopee_net = ImageMetricLearning(
        param=param, backbone=backbone, head=head, metric=metric, match=match
    )
    shopee_net.to(get_device())

    return shopee_net


# %%
def create_trainer(config: Any) -> ShopeeTrainer:
    #trainer = ShopeeTrainer(config, ckpt_filename_base="exp-011")
    trainer = ShopeeTrainer(config, ckpt_filename_base="exp-011", monitor="valid_f1", mode="max")
    return trainer


# %%
def train() -> None:
    config_defaults = get_config_defaults()
    with context(config_defaults, JobType.Training) as config:
        dm = create_datamodule(config)
        system = create_system(config)
        trainer = create_trainer(config)

        trainer.fit(system, datamodule=dm)
        trainer.save_best_model()


if __name__ == "__main__":
    from shopee_product_matching import agent

    #agent.run(train)
    train()