import pandas as pd

from shopee_product_matching.feature import find_matches, TfIdfEmbedding, LaserEmbedding
from shopee_product_matching.constants import Paths
from shopee_product_matching.util import save_submission_csv, ensemble

#df = pd.read_csv(Paths.shopee_product_matching / "train.csv")
df = pd.read_csv(Paths.shopee_product_matching / "test.csv")
posting_ids = df["posting_id"].to_list()

with ensemble():
    ind_laser_dir = Paths.requirements / "LASER"
    ind_laser_model = LaserEmbedding("id", laser_dir=ind_laser_dir)
    ind_laser_embeddings = ind_laser_model.transform(df["title"])
    ind_laser_preds = find_matches(
        posting_ids=posting_ids, embeddings=ind_laser_embeddings, threshold=0.25
    )
    save_submission_csv(posting_ids, ind_laser_preds, "submission_ind_laser.csv")

    tfidf_model = TfIdfEmbedding()
    tfidf_embeddings = tfidf_model.fit_transform(df["title"])
    tfidf_preds = find_matches(
        posting_ids=posting_ids, embeddings=tfidf_embeddings, threshold=0.6
    )
    save_submission_csv(posting_ids, tfidf_preds, "submission_tfidf.csv")