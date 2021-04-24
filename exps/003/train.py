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
#%load_ext autoreload
#%autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
from shopee_product_matching import constants, storage
from shopee_product_matching.logger import get_logger
from shopee_product_matching.metric import ArcMarginProduct
from shopee_product_matching.system import TitleMetricLearning
from shopee_product_matching.util import (JobType, finalize, get_device, string_escape,
                                          get_params_by_inspection, initialize)
from shopee_product_matching.feature import TfIdfEmbedding
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
initialize(SEED, JobType.Training, params=get_params_by_inspection())

# %%
shopee_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(title=True, label_group=True),
    valid_query=ShopeeQuery(title=True, label_group=True),
    test_query=ShopeeQuery(title=True),
)
shopee_dm = ShopeeDataModule(shopee_dm_param)
shopee_dm.setup()
# %%
tfidf_embeddings = TfIdfEmbedding(max_features=TFIDF_MAX_FEATURES, n_components=PCA_N_COMPONENTS)
tfidf_embeddings.fit(shopee_dm.train_dataset.df["title"])
tfidf_model_path = tfidf_embeddings.save("tfidf.model")
storage.save(tfidf_model_path)

# %%
def title_embedding(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = df["title"].apply(string_escape)
    list_vecs = tfidf_embeddings(df["title"]).tolist()
    df["title"] = pd.Series(list_vecs, index=df.index)
    return df
shopee_dm.setup(train_df_preproc=title_embedding, valid_df_preproc=title_embedding)

# %%
from shopee_product_matching.network import AffineHead
import torch.nn as nn

head_layers = [AffineHead(tfidf_embeddings.num_features, out_dim=NUM_FEATURES)]
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