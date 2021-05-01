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
from pathlib import Path
from typing import Dict, Any

import torch.nn
import pytorch_lightning as pl
import timm
from shopee_product_matching.transform import read_resize_normalize, identity
from shopee_product_matching import constants
from shopee_product_matching.metric import create_metric
from shopee_product_matching.trainer import ShopeeTrainer
from shopee_product_matching.system import ImageMetricLearning
from shopee_product_matching.datamodule import (
    ShopeeDataModule,
    ShopeeDataModuleQueries,
    ShopeeQuery,
)
from shopee_product_matching.util import (
    JobType,
    context,
    ensemble,
    get_device,
    is_kaggle,
    get_model_path,
)
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
from shopee_product_matching.transform import imread, identity, map_stack



# %%
if is_kaggle():
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()

# %%
def get_config_defaults() -> Dict[str, Any]:
    return {
        "test_batch_size": 64,
        "num_workers": 4,
        "image_size": 512,
        "backbone": "efficientnet_b3",
        "metric": "arcface",
        "checkpoint_filenames": [
            "exp-005-effb3/exp-005-fold=1-epoch=7-val_loss=0.00.ckpt",
            "exp-005-effb3/exp-005-fold=4-epoch=4-val_loss=0.00.ckpt",
            "exp-005-effb3/exp-005-fold=2-epoch=7-val_loss=0.00.ckpt",
            "exp-005-effb3/exp-005-fold=3-epoch=7-val_loss=0.00.ckpt",
            "exp-005-effb3/exp-005-fold=0-epoch=7-val_loss=0.00.ckpt",
        ],
    }


# %%

def create_datamodule(config: Any) -> ShopeeDataModule:
    image_transform = Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomApply(torch.nn.ModuleList([RandomRotation(degrees=120)]), p=0.8),
            RandomApply(
                torch.nn.ModuleList([ColorJitter(brightness=(0.09, 0.6))]), p=0.5
            ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    queries = ShopeeDataModuleQueries(
        test=ShopeeQuery(
            image=Compose(
                [
                    imread("test_images"),
                    Resize(size=(config.image_size, config.image_size)),
                    map_stack(
                        [
                            image_transform,
                            image_transform,
                            image_transform,
                            image_transform,
                            image_transform,
                        ]
                    ),
                ]
            ),
            posting_id=identity
        )
    )
    return ShopeeDataModule(config, queries)


# %%
def create_system(config: Any, checkpoint_filename: str) -> pl.LightningModule:
    tag = Path(checkpoint_filename).stem
    backbone = timm.create_model(
        config.backbone, pretrained=True, num_classes=0, global_pool=""
    )
    metric = create_metric(
        config.metric,
        num_features=backbone.num_features,
        num_classes=constants.TrainData.label_group_unique_unique_count,
        s=30.0,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
    )
    shopee_net = ImageMetricLearning.load_from_checkpoint(
        str(get_model_path(checkpoint_filename)),
        backbone=backbone,
        metric=metric,
        submission_filename=f"submission_{tag}.csv",
    )
    shopee_net.to(get_device())

    return shopee_net


# %%
def create_trainer(config: Any) -> ShopeeTrainer:
    trainer = ShopeeTrainer(config, monitor=None)
    return trainer


# %%
def infer() -> None:
    config_defaults = get_config_defaults()
    with context(config_defaults, JobType.Inferene) as config:
        dm = create_datamodule(config)
        trainer = create_trainer(config)

        with ensemble():
            for checkpoint_filename in config.checkpoint_filenames:
                system = create_system(config, checkpoint_filename)
                trainer.test(system, datamodule=dm)


# %%
if __name__ == "__main__":
    infer()