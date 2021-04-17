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
import timm
from shopee_product_matching.util import (
    pass_as_image,
    get_params_by_inspection,
    initialize,
    finalize,
    get_device,
    JobType,
)
from shopee_product_matching.metric import ArcMarginProduct
from shopee_product_matching.system import ImageMetricLearning
from shopee_product_matching.logger import get_logger
from shopee_product_matching import storage, constants
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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

# %%
initialize(SEED, JobType.Training, params=get_params_by_inspection())

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
from shopee_product_matching.datamodule import (
    ShopeeDataModule,
    ShopeeDataModuleParam,
    ShopeeQuery,
)

shopee_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(image=pass_as_image(train_transform), label_group=True),
    valid_query=ShopeeQuery(image=pass_as_image(valid_transform), label_group=True),
    test_query=ShopeeQuery(image=pass_as_image(valid_transform)),
)
shopee_dm = ShopeeDataModule(shopee_dm_param)

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
shopee_net = ImageMetricLearning(param=param, backbone=backbone, metric=metric)
shopee_net.to(get_device())

# %%
checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    dirpath="checkpoints",
    filename="exp-001-{epoch:02d}-{valid_loss:.2f}",
    save_top_k=3,
    mode="min",
)
early_stop_callback = EarlyStopping(
    patience=EARLY_STOP_PATIENCE, verbose=True, monitor="valid_loss", mode="min"
)

# %%
trainer = pl.Trainer(
    precision=16,
    gpus=1,
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=MAX_EPOCHS,
    overfit_batches=OVERFIT_BATCHES,
    logger=get_logger(),
)
trainer.fit(shopee_net, shopee_dm)

# %%
storage.save(checkpoint_callback.best_model_path)
# %%
finalize()