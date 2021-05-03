import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


from shopee_product_matching.constants import Paths
from shopee_product_matching.neighbor import KnnMatch
from shopee_product_matching.metric import f1_score
from shopee_product_matching.util import get_matches
from shopee_product_matching.feature import FastTextEmbedding, TfIdfEmbedding

df = pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0)

tfidf_model = TfIdfEmbedding()
tfidf_model.fit(df["title"])

fasttext_model = FastTextEmbedding(
    dim=300,
    epoch=128,
    #model="cbow",
    pretrained_vectors="cc.id.300.vec",
    model="skipgram",
    #pretrained_vectors="wiki.id.vec",
    agg_func=FastTextEmbedding.create_tfidf_agg_func(tfidf_model),
)
fasttext_embeddings = fasttext_model.fit_transform(df["title"])


def doit(th):
    indices = KnnMatch(threshold=th)(fasttext_embeddings)
    infer_matches = [[df.index[i] for i in match] for match in indices]

    exp = get_matches(posting_ids=df.index.values, label_groups=df["label_group"])
    f1 = f1_score(infer_matches, exp)
    print(f1)

print([doit(i * 0.5) for i in range(20)])