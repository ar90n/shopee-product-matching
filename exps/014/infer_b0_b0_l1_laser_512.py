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

import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import timm
from shopee_product_matching.network import AffineHead
from shopee_product_matching.tasks import test_tfidf, test_laser
from shopee_product_matching.transform import read_resize_normalize, identity
from shopee_product_matching.constants import TrainData
from shopee_product_matching.metric import create_metric
from shopee_product_matching.neighbor import create_match
from shopee_product_matching.trainer import ShopeeTrainer
from shopee_product_matching.system import ImageMetricLearning
from shopee_product_matching.datamodule import (
    ShopeeDataModule,
    ShopeeDataModuleQueries,
    ShopeeQuery,
    ShopeeDataModuleTransforms,
)
from shopee_product_matching.util import (
    JobType,
    context,
    ensemble,
    get_device,
    is_kaggle,
    get_model_path,
    save_submission_confidence,
    string_escape,
)

# %%
if is_kaggle():
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()

# %%
def get_config_defaults() -> Dict[str, Any]:
    return {
        "is_cv": False,
        "test_batch_size": 16,
        "num_workers": 4,
        "image_size": 512,
        "model_params": [
            (
                "efficientnet_b0",
                "exp-014-effb0-512/exp-014-fold0-epoch11-val_loss0.00.ckpt",
                {"name": "affine", "dim": 512},
                {
                    "name": "arcface",
                    "num_classes": TrainData.label_group_unique_count_pdf_fold,
                },
                {"name": "cosine", "threshold": 0.45},
            ),
            (
                "efficientnet_b0",
                "exp-014-effb0-512/exp-014-fold1-epoch10-val_loss0.00.ckpt",
                {"name": "affine", "dim": 512},
                {
                    "name": "arcface",
                    "num_classes": TrainData.label_group_unique_count_pdf_fold,
                },
                {"name": "cosine", "threshold": 0.45},
            ),
            (
                "eca_nfnet_l1",
                "exp-014-nfnet-l1/exp-014-fold4-epoch11-val_loss0.00-v1.ckpt",
                {"name": "affine", "dim": 1024},
                {
                    "name": "arcface",
                    "num_classes": TrainData.label_group_unique_count_pdf_fold,
                },
                {"name": "cosine", "threshold": 0.45},
            ),
        ],
    }


# %%
def create_datamodule(config: Any) -> ShopeeDataModule:
    queries = ShopeeDataModuleQueries(
        test=ShopeeQuery(image=read_resize_normalize(config), posting_id=identity),
    )

    def _escape(df: pd.DataFrame) -> pd.DataFrame:
        df["title"] = df["title"].apply(string_escape)
        return df

    transforms = ShopeeDataModuleTransforms(test=_escape)

    dm = ShopeeDataModule(config, queries, transforms=transforms)
    dm.setup()
    return dm


# %%
def create_system(
    config: Any,
    model_param: Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]],
) -> pl.LightningModule:
    backbone_name, checkpoint_filename, head_info, metric_info, match_info = model_param
    tag = Path(checkpoint_filename).stem
    backbone = timm.create_model(
        backbone_name, pretrained=True, num_classes=0, global_pool=""
    )
    if head_info["name"] == "affine":
        head = AffineHead(in_dim=backbone.num_features, out_dim=head_info["dim"])
        num_features = head_info["dim"]
    else:
        head = nn.Identity()
        num_features = backbone.num_features
    metric = create_metric(num_features=num_features, **metric_info)
    match = create_match(**match_info)
    shopee_net = ImageMetricLearning.load_from_checkpoint(
        str(get_model_path(checkpoint_filename)),
        backbone=backbone,
        head=head,
        metric=metric,
        match=match,
        submission_filename=f"submission_{tag}.csv",
        save_submission_confidence=True,
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

        df = dm._test_dataset.df
        with ensemble(conf_threshold=0.65):
            for model_param in config.model_params:
                system = create_system(config, model_param)
                trainer.test(system, datamodule=dm)

            test_tfidf.main(df, param={"max_features": 20000, "threshold": 0.20})
            test_laser.main(df, param={"threshold": 0.08})


# %%
if __name__ == "__main__":
    infer()
