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
from shopee_product_matching.metric import f1_score
from shopee_product_matching.util import (JobType, finalize,
                                          get_params_by_inspection,
                                          initialize)

## %%
#%load_ext autoreload
#%autoreload 2

# %% tags=["parameters"]
SEED = 42
THRESHOLD = 0.6

# %%
initialize(SEED, JobType.Inferene, params=get_params_by_inspection())
# %%
from shopee_product_matching.datamodule import (ShopeeDataModule,
                                                ShopeeDataModuleParam,
                                                ShopeeQuery)

shopee_dm_param = ShopeeDataModuleParam(
    train_query=ShopeeQuery(
        title=True, label_group=True
    ),
    valid_query=ShopeeQuery(
        title=True, label_group=True
    ),
    test_query=ShopeeQuery(title=True),
)
shopee_dm = ShopeeDataModule(shopee_dm_param)
shopee_dm.setup()
# %%
df = shopee_dm.train_dataset.df

# %%
titles = df["title"].tolist()
expect_matches = df["label_group"].map(df.groupby(["label_group"])["posting_id"].unique().to_dict())
# %%
from shopee_product_matching.feature import TfIdfEmbedding, find_matches
embeddings = TfIdfEmbedding().fit_transform(titles)
posting_ids = df["posting_id"].tolist()
# %%
text_matches = find_matches(posting_ids, embeddings, THRESHOLD)
print(f1_score(text_matches, expect_matches))

# %%
finalize()