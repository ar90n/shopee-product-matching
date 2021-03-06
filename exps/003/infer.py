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
import pickle
import hashlib
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from shopee_product_matching.network import AffineHead
from shopee_product_matching import constants, util
from shopee_product_matching.logger import get_logger
from shopee_product_matching.metric import ArcMarginProduct
from shopee_product_matching.system import TitleMetricLearning
from shopee_product_matching.util import (
    JobType,
    finalize,
    get_params_by_inspection,
    get_requirements,
    initialize,
)

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
EARLY_STOP_PATIENCE = 5
TITLE_MODEL_CHECKPOINT_FILE_NAME: str = (
    "exp-003-train-one-fc-epoch05-valid_loss14.09.ckpt"
)
TITLE_TFIDF_MODEL_FILE_NAME: str = "tfidf-max_feature15000-n_component5000.model"
NUM_FEATURES = 500
TFIDF_MAX_FEATURES = 15000
PCA_N_COMPONENTS = 5000

# %%
initialize(SEED, JobType.Inferene, params=get_params_by_inspection())
# %%
from shopee_product_matching.tasks import transform_tfidf

tfidf_model_path = (
    util.get_requirements() / TITLE_TFIDF_MODEL_FILE_NAME
)
test_df = pd.read_csv(util.get_input() / "shopee-product-matching" / "test.csv")
tfidf_embedding_path = transform_tfidf.main(tfidf_model_path, test_df["title"])

# %%
with open(tfidf_embedding_path, "rb") as fp:
    tfidf_embeddings = pickle.load(fp)


def title_embedding(title: str) -> np.ndarray:
    hash = hashlib.sha224(title.encode()).hexdigest()
    return tfidf_embeddings[hash]


# %%
from shopee_product_matching.datamodule import (
    ShopeeDataModule,
    ShopeeDataModuleParam,
    ShopeeQuery,
)

title_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(),
    valid_query=ShopeeQuery(),
    test_query=ShopeeQuery(title=title_embedding),
)
title_dm = ShopeeDataModule(title_dm_param)

# %%
import torch.nn as nn
head = nn.Sequential(*[AffineHead(PCA_N_COMPONENTS, out_dim=NUM_FEATURES)])
metric = ArcMarginProduct(
    NUM_FEATURES,
    constants.TrainData.label_group_unique_unique_count,
    s=30.0,
    m=0.50,
    easy_margin=False,
    ls_eps=0.0,
)
param = TitleMetricLearning.Param(max_lr=1e-5 * TRAIN_BATCH_SIZE)
# %%
title_net = TitleMetricLearning.load_from_checkpoint(
    str(get_requirements() / TITLE_MODEL_CHECKPOINT_FILE_NAME),
    param=param,
    head=head,
    metric=metric,
    submission_filename="submission.csv",
)

# %%
trainer = pl.Trainer(
    precision=16,
    gpus=1,
    logger=get_logger(),
)
trainer.test(title_net, datamodule=title_dm)

# %%
util.clean_up()


# %%
finalize()
