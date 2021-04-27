import hashlib
import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import pandas as pd

from shopee_product_matching.feature import TfIdfEmbedding


def main(
    model_path: Path, titles: pd.Series, output_filename: Optional[str] = None
) -> Path:
    obj = {
        "model_path": model_path,
        "titles": titles,
        "output_filename": output_filename,
    }

    with TemporaryDirectory() as temp:
        input_path = Path(temp) / "input.pickle"
        output_path = Path(temp) / "output.pickle"
        with input_path.open("wb") as fp:
            pickle.dump(obj, fp)
        subprocess.check_call(
            [
                sys.executable,
                __file__,
                str(input_path.absolute()),
                str(output_path.absolute()),
            ]
        )

        with output_path.open("rb") as fp:
            return pickle.load(fp)


def _main(model_path: Path, titles: pd.Series, output_filename: Optional[str]) -> Path:
    tfidf_embedding = TfIdfEmbedding.load(model_path)
    embeddings = tfidf_embedding(titles)
    result = {
        hashlib.sha224(title.encode()).hexdigest(): feature
        for title, feature in zip(titles, embeddings)
    }

    output_filename = (
        "tfidf_embeddings.pickle" if output_filename is None else output_filename
    )
    output_path = Path().cwd() / output_filename
    with output_path.open("wb") as fp:
        pickle.dump(result, fp)

    return output_path


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as fp:
        obj = pickle.load(fp)
    ret = _main(**obj)
    with open(sys.argv[2], "wb") as fp:
        pickle.dump(ret, fp)
