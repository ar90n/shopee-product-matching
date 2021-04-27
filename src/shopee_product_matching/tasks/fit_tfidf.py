import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import pandas as pd

from shopee_product_matching.feature import TfIdfEmbedding


def main(
    titles: pd.Series,
    param: Optional[Dict[str, Any]] = None,
    output_filename: Optional[str] = None,
) -> Path:
    obj = {"titles": titles, "param": param, "output_filename": output_filename}

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


def _main(
    titles: pd.Series,
    param: Optional[Dict[str, Any]] = None,
    output_filename: Optional[str] = None,
) -> Path:
    param = {} if param is None else param
    tfidf_embedding = TfIdfEmbedding()
    tfidf_embedding.fit(titles)

    output_filename = "tfidf.model" if output_filename is None else output_filename
    return tfidf_embedding.save(output_filename)


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as fp:
        obj = pickle.load(fp)
    ret = _main(**obj)
    with open(sys.argv[2], "wb") as fp:
        pickle.dump(ret, fp)
