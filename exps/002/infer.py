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
import pytorch_lightning as pl
import timm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from shopee_product_matching import constants, storage, util
from shopee_product_matching.logger import get_logger
from shopee_product_matching.metric import ArcMarginProduct
from shopee_product_matching.system import ImageMetricLearning
from shopee_product_matching.util import (JobType, finalize,
                                          get_params_by_inspection,
                                          get_requirements, initialize,
                                          is_kaggle, pass_as_image)

# %%
# %load_ext autoreload
# %autoreload 2

# %% tags=["parameters"]
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 5
OVERFIT_BATCHES = 1.0
SEED = 42
BACKBONE = "efficientnet_b3"
EARLY_STOP_PATIENCE = 5
DIM = [512, 512]
CHECKPOINT_FILE_NAME: str = "exp-001-epoch17-valid_loss1.02.ckpt"

# %%
initialize(SEED, JobType.Inferene, params=get_params_by_inspection())
if is_kaggle():
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()
# %%
import albumentations

train_transform = albumentations.Compose(
    [
        albumentations.Resize(DIM[0], DIM[1], always_apply=True),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(limit=120, p=0.8),
        albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        albumentations.Normalize(),
    ]
)
valid_transform = albumentations.Compose(
    [
        albumentations.Resize(DIM[0], DIM[1], always_apply=True),
        albumentations.Normalize(),
    ]
)

# %%
from shopee_product_matching.datamodule import (ShopeeDataModule,
                                                ShopeeDataModuleParam,
                                                ShopeeQuery)

shopee_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(
        image=pass_as_image(train_transform), title=True, label_group=True
    ),
    valid_query=ShopeeQuery(
        image=pass_as_image(valid_transform), title=True, label_group=True
    ),
    test_query=ShopeeQuery(image=pass_as_image(valid_transform), title=True),
)
shopee_dm = ShopeeDataModule(shopee_dm_param)
shopee_dm.setup()

# %%
from shopee_product_matching.feature import TfIdfEmbedding, find_matches
from shopee_product_matching.util import save_submission_csv

em = TfIdfEmbedding().fit_transform(shopee_dm.test_dataset.df["title"].tolist())
posting_ids = shopee_dm.test_dataset.df["posting_id"].tolist()
text_matches = find_matches(posting_ids, em, 0.6)
save_submission_csv(posting_ids, text_matches, "submission_text.csv")
# %%
del em, text_matches
util.clean_up()

# %%
backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0, global_pool="")
metric = ArcMarginProduct(
    backbone.num_features,
    constants.TrainData.label_group_unique_unique_count,
    s=30.0,
    m=0.50,
    easy_margin=False,
    ls_eps=0.0,
)
param = ImageMetricLearning.Param(max_lr=1e-5 * TRAIN_BATCH_SIZE)
shopee_net = ImageMetricLearning.load_from_checkpoint(
    str(get_requirements() / CHECKPOINT_FILE_NAME),
    param=param,
    backbone=backbone,
    metric=metric,
    submission_filename="submission_image.csv",
)

# %%
trainer = pl.Trainer(
    precision=16,
    gpus=1,
    logger=get_logger(),
)
trainer.test(shopee_net, datamodule=shopee_dm)

# %%
util.clean_up()
# %%
import pandas as pd

submission_text = pd.read_csv("submission_text.csv", index_col=0).applymap(
    lambda x: x.split(" ")
)
submission_image = pd.read_csv("submission_image.csv", index_col=0).applymap(
    lambda x: x.split(" ")
)
submission_ensembled = (
    (submission_image + submission_text).applymap(set).applymap(lambda x: " ".join(x))
)
submission_ensembled.to_csv("submission.csv")

# %%
finalize()
