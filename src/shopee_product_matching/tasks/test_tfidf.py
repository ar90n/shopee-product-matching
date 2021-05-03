import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import pandas as pd
from shopee_product_matching.feature import TfIdfEmbedding, find_matches
from shopee_product_matching.util import save_submission_csv


def main(
    df: pd.DataFrame,
    param: Optional[Dict[str, Any]] = None,
) -> None:
    obj = {"df": df, "param": param}

    _main(df, param)
    #with TemporaryDirectory() as temp:

    #    input_path = Path(temp) / "input.pickle"
    #    output_path = Path(temp) / "output.pickle"
    #    with input_path.open("wb") as fp:
    #        pickle.dump(obj, fp)
    #    subprocess.check_call(
    #        [
    #            sys.executable,
    #            __file__,
    #            str(input_path.absolute()),
    #            str(output_path.absolute()),
    #        ]
    #    )


def _main(
    df: pd.DataFrame,
    param: Optional[Dict[str, Any]] = None,
) -> None:
    param = {} if param is None else param

    tfidf_embedding_params = {}
    if "max_featreus" in param:
        tfidf_embedding_params["max_features"] = param["max_fetures"]
    if "n_components" in param:
        tfidf_embedding_params["n_components"] = param["n_components"]

    match_params = {}
    if "threshold" in param:
        match_params["threshold"] = param["threshold"]

    posting_ids = df["posting_id"].to_list()
    tfidf_model = TfIdfEmbedding(**tfidf_embedding_params)
    tfidf_embeddings = tfidf_model.fit_transform(df["title"])
    tfidf_preds = find_matches( posting_ids=posting_ids, embeddings=tfidf_embeddings, **match_params)
    save_submission_csv(posting_ids, tfidf_preds, "submission_tfidf.csv")

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as fp:
        obj = pickle.load(fp)
    _main(**obj)