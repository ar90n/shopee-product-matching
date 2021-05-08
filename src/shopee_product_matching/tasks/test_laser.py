import pandas as pd
from typing import Dict, Any

from shopee_product_matching.constants import Paths
from shopee_product_matching.feature import LaserEmbedding, find_matches
from shopee_product_matching.util import (
    save_submission_csv,
    save_submission_confidence,
)
from shopee_product_matching.neighbor import CosineSimilarityMatch, KnnMatch


def main(
    df: pd.DataFrame,
    param: Dict[str, Any],
    save_submission_confidence=False,
    weight=1.0,
) -> None:
    posting_ids = df["posting_id"].to_list()
    threshold = param.get("threshold", 0.1)
    ind_laser_dir = Paths.requirements / "LASER"
    ind_laser_model = LaserEmbedding("id", laser_dir=ind_laser_dir)
    ind_laser_embeddings = ind_laser_model.transform(df["title"])

    if save_submission_confidence:
        save_submission_confidence(
            posting_ids,
            ind_laser_embeddings,
            threshold,
            "submission_ind_laser.csv",
            weight,
        )
    else:
        ind_laser_preds = find_matches(
            posting_ids=posting_ids,
            embeddings=ind_laser_embeddings,
            matcher=CosineSimilarityMatch(threshold=threshold),
        )
        save_submission_csv(posting_ids, ind_laser_preds, "submission_ind_laser.csv")