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
import hashlib
import numpy as np
import pickle
import pandas as pd
from numpy.core.defchararray import title
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
from shopee_product_matching import constants, storage
from shopee_product_matching import util
from shopee_product_matching.logger import get_logger
from shopee_product_matching.metric import ArcMarginProduct
from shopee_product_matching.system import TitleMetricLearning
from shopee_product_matching.util import (JobType, finalize, get_device, string_escape,
                                          get_params_by_inspection, initialize)
from shopee_product_matching.datamodule import (ShopeeDataModule,
                                                ShopeeDataModuleParam,
                                                ShopeeQuery)

# %% tags=["parameters"]
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 5
OVERFIT_BATCHES = 0
FAST_DEV_RUN = False
SEED = 42
EARLY_STOP_PATIENCE = 5
NUM_FEATURES = 500
TFIDF_MAX_FEATURES = 15000
PCA_N_COMPONENTS = 5000
HEAD_LAYERS = 1

# %%
shopee_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(),
    valid_query=ShopeeQuery(),
    test_query=ShopeeQuery(),
)
shopee_dm = ShopeeDataModule(shopee_dm_param)
shopee_dm.setup()
titles = shopee_dm.train_dataset.df["title"]
titles = titles.apply(string_escape)

# %%
from shopee_product_matching.tasks import fit_tfidf
tfidf_model_path = fit_tfidf.main(titles)


# %%
train_all_df = pd.read_csv(util.get_input() / "shopee-product-matching" / "train.csv")
titles_all = train_all_df["title"]

# %%
from shopee_product_matching.tasks import transform_tfidf
tfidf_embedding_path = transform_tfidf.main(tfidf_model_path, titles_all)
# %%
initialize(SEED, JobType.Training, params=get_params_by_inspection())

# %%
storage.save(str(tfidf_model_path))
# %%
with open(tfidf_embedding_path, "rb") as fp:
    tfidf_embeddings = pickle.load(fp)
def title_embedding(title: str) -> np.ndarray:
    hash = hashlib.sha224(title.encode()).hexdigest()
    return tfidf_embeddings[hash]
# %%
shopee_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(title=title_embedding, label_group=True),
    valid_query=ShopeeQuery(title=title_embedding, label_group=True),
    test_query=ShopeeQuery(),
)
shopee_dm = ShopeeDataModule(shopee_dm_param)
shopee_dm.setup()

# %%
from shopee_product_matching.network import AffineHead
import torch.nn as nn

head_layers = [AffineHead(PCA_N_COMPONENTS, out_dim=NUM_FEATURES)]
for _ in range(HEAD_LAYERS - 1):
    head_layers.append(AffineHead(NUM_FEATURES, out_dim=NUM_FEATURES))
head = nn.Sequential(*head_layers)

# %%
metric = ArcMarginProduct(
    NUM_FEATURES,
    constants.TrainData.label_group_unique_unique_count,
    s=30.0,
    m=0.50,
    easy_margin=False,
    ls_eps=0.0,
)
# %%
param = TitleMetricLearning.Param(max_lr=1e-5 * TRAIN_BATCH_SIZE)
shopee_net = TitleMetricLearning(param=param, head=head, metric=metric)
shopee_net.to(get_device())

# %%
checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    dirpath="checkpoints",
    filename="exp-003-train-one-fc-{epoch:02d}-{valid_loss:.2f}",
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
    fast_dev_run=FAST_DEV_RUN,
    logger=get_logger(),
)
trainer.fit(shopee_net, datamodule=shopee_dm)

# %%
storage.save(checkpoint_callback.best_model_path)
# %%
finalize()