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
from typing import Dict, Any, Tuple

import pandas as pd
import pytorch_lightning as pl
import timm
from shopee_product_matching.transform import read_resize_normalize, identity
from shopee_product_matching import constants
from shopee_product_matching.neighbor import create_match
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
    save_submission_embedding,
)

# %%
if is_kaggle():
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()

# %%
def get_config_defaults() -> Dict[str, Any]:
    return {
        "is_cv": True,
        "test_batch_size": 16,
        "num_workers": 4,
        "image_size": 512,
        "backbone": "efficientnet_b0",
        "metric": "arcface",
        "checkpoint_filenames": [
            "exp-005-effb0/exp-005-fold0-epoch5-val_loss0.00.ckpt",
            "exp-005-effb0/exp-005-fold1-epoch6-val_loss0.00.ckpt",
            "exp-005-effb0/exp-005-fold2-epoch8-val_loss0.00.ckpt",
            "exp-005-effb0/exp-005-fold3-epoch8-val_loss0.00.ckpt",
            "exp-005-effb0/exp-005-fold4-epoch7-val_loss0.00.ckpt",
        ],
    }


# %%
def create_datamodule(config: Any, fold: int) -> ShopeeDataModule:
    queries = ShopeeDataModuleQueries(
        test=ShopeeQuery(image=read_resize_normalize(config), posting_id=identity),
    )

    def test_valid_split(df: pd.DataFrame, _: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from shopee_product_matching.constants import Paths

        fold_df = pd.read_csv(Paths.requirements / "fold.csv", index_col=0)
        train_df = df[fold_df["fold"] != fold]
        valid_df = df[fold_df["fold"] == fold]
        return train_df, valid_df

    return ShopeeDataModule(config, queries, test_valid_split)


# %%
def create_system(config: Any, checkpoint_filename: str) -> pl.LightningModule:
    tag = Path(checkpoint_filename).stem
    backbone = timm.create_model(
        config.backbone, pretrained=True, num_classes=0, global_pool=""
    )
    metric = create_metric(
        config.metric,
        num_features=backbone.num_features,
        #num_classes=constants.TrainData.label_group_unique_count_pdf_fold,
        num_classes=constants.TrainData.label_group_unique_unique_count,
    )
    match = create_match("cosine", threshold=0.45)
    shopee_net = ImageMetricLearning.load_from_checkpoint(
        str(get_model_path(checkpoint_filename)),
        backbone=backbone,
        metric=metric,
        match=match,
        save_submission_embedding=True,
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
        trainer = create_trainer(config)

        with ensemble():
            for i, checkpoint_filename in enumerate(config.checkpoint_filenames):
                dm = create_datamodule(config, i)
                system = create_system(config, checkpoint_filename)
                trainer.test(system, datamodule=dm)


# %%
if __name__ == "__main__":
    infer()