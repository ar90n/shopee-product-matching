import hashlib
import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from shopee_product_matching.feature import LaserEmbedding
from shopee_product_matching.constants import Paths

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def main(
    titles: pd.Series, output_filename: Optional[str] = None
) -> Dict[str, np.ndarray]:
    obj = {
        "titles": titles,
        "output_filename": output_filename,
    }
    print(titles)

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
            result_path = pickle.load(fp)
        with result_path.open("rb") as fp:
            return pickle.load(fp)


def _main(titles: pd.Series, output_filename: Optional[str]) -> Path:
    ind_laser_dir = Paths.requirements / "LASER"
    model = LaserEmbedding("id", laser_dir=ind_laser_dir)
    embeddings = model.transform(titles)
    result = {
        hashlib.sha224(title.encode()).hexdigest(): feature
        for title, feature in zip(titles, embeddings)
    }

    output_filename = (
        "laser_embeddings.pickle" if output_filename is None else output_filename
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
