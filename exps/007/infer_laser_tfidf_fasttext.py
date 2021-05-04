import fasttext
import pandas as pd

from shopee_product_matching.feature import (
    find_matches,
    TfIdfEmbedding,
    LaserEmbedding,
    FastTextEmbedding,
)
from shopee_product_matching.constants import Paths
from shopee_product_matching.util import (
    save_submission_csv,
    ensemble,
    get_matches,
    get_model_path,
)
from shopee_product_matching.metric import f1_score
from shopee_product_matching.neighbor import KnnMatch, CosineSimilarityMatch

get_cv = False
if get_cv:
    df = pd.read_csv(Paths.shopee_product_matching / "train.csv")
else:
    df = pd.read_csv(Paths.shopee_product_matching / "test.csv")
posting_ids = df["posting_id"].to_list()

with ensemble():
    ind_laser_dir = Paths.requirements / "LASER"
    ind_laser_model = LaserEmbedding("id", laser_dir=ind_laser_dir, chunk_size=8192)
    ind_laser_embeddings = ind_laser_model.transform(df["title"])
    ind_laser_preds = find_matches(
        posting_ids=posting_ids,
        embeddings=ind_laser_embeddings,
        matcher=KnnMatch(threshold=0.22),
    )
    save_submission_csv(posting_ids, ind_laser_preds, "submission_ind_laser.csv")

    tfidf_model = TfIdfEmbedding()
    tfidf_embeddings = tfidf_model.fit_transform(df["title"])
    tfidf_preds = find_matches(
        posting_ids=posting_ids,
        embeddings=tfidf_embeddings,
        matcher=CosineSimilarityMatch(threshold=0.33),
    )
    save_submission_csv(posting_ids, tfidf_preds, "submission_tfidf.csv")

    fasttext_model = FastTextEmbedding(
        dim=32,
        epoch=32,
        model="skipgram",
        min_count=3,
        #pretrained_vectors=get_model_path("cc.id.300.vec"),
        agg_func=FastTextEmbedding.create_tfidf_agg_func(tfidf_model),
    )
    fasttext_embeddings = fasttext_model.fit_transform(df["title"])
    fasttext_preds = find_matches(
        posting_ids=posting_ids,
        embeddings=fasttext_embeddings,
        matcher=KnnMatch(threshold=0.4),
    )
    save_submission_csv(posting_ids, fasttext_preds, "submission_fasttext.csv")

if get_cv:
    submission = pd.read_csv("submission.csv", index_col=0)
    submission_matches = submission["matches"].map(lambda x: x.split(" "))
    exp_df = pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0).loc[
        submission.index
    ]
    exp_matches = get_matches(
        exp_df.index,
        exp_df["label_group"],
    )

    f1 = f1_score(submission_matches, exp_matches)
    print(f"f1:{f1}")