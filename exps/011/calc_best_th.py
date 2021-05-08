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
import sys
from pathlib import Path
import torch
import pandas as pd
from torch.nn.functional import embedding_bag, threshold

from shopee_product_matching.constants import Paths
from shopee_product_matching.util import get_matches
from shopee_product_matching.neighbor import CosineSimilarityMatch, KnnMatch

# %%
#tensor_path = Path.cwd() / "submission_effv2s.pt"
#tensor_path = Path.cwd() / "submission_nf_regnet_b1.pt"
tensor_path = Path(sys.argv[1])
#submission_path = Path.cwd() / "submission.csv"
#submission_path = Path.cwd() / "submission_nf_regnet_b1.csv"
submission_path = Path(sys.argv[2])
# %%
embedding = torch.load(tensor_path)
submission = pd.read_csv(submission_path, index_col=0)
# %%
train_df = pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0)
fold_df = pd.read_csv(Paths.requirements / "fold.csv", index_col=0)[["fold"]]
df = submission.join(train_df).join(fold_df).reset_index()
# %%
# %%
from shopee_product_matching.metric import f1_score
def calc_f1_score(df, embedding, matcher):
    expect_matches = get_matches(
        df.index, df["label_group"]
    )
    index_matches = matcher(embedding[df.index,:])
    infer_matches = [
        [df.index.values[i] for i in match]
        for match in index_matches
    ]
    return f1_score(infer_matches, expect_matches)
# %%
ret = []
for i in range(5):
    valid_df = df[df["fold"] < (i+1)]
    for j in range(7):
        threshold = 0.1 * (j + 1)
        #matcher = CosineSimilarityMatch(threshold=threshold)
        matcher = KnnMatch(threshold=threshold, metric="cosine")
        f1 = calc_f1_score(valid_df, embedding, matcher)
        ret.append((f1, threshold))
# %%
print(ret)
# %%
print(max(ret))