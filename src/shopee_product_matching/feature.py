import os
from pathlib import Path
from typing import Iterable, List, Union, Optional
from tempfile import TemporaryDirectory
import subprocess

import cudf
import numpy as np
import pandas as pd
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from joblib import dump, load
from sklearn.decomposition import PCA
from torch.nn.functional import embedding


class LaserEmbedding:
    def __init__(self, lang: str, laser_dir: Optional[Path] = None) -> None:
        self._lang = lang
        self._laser_dir = laser_dir

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        laser_dir = (
            self._laser_dir if self._laser_dir is not None else os.environ.get("LASER")
        )
        if laser_dir is None:
            raise EnvironmentError("Path to LASER directory is not given")

        embed_sh_path = Path(laser_dir) / "tasks" / "embed" / "embed.sh"
        with TemporaryDirectory() as temp:
            input_path = Path(temp) / "input.txt"
            output_path = Path(temp) / "output.raw"
            input_path.write_text("\n".join(texts))

            subprocess.check_call(
                args=[
                    "bash",
                    str(embed_sh_path),
                    str(input_path),
                    self._lang,
                    str(output_path),
                ],
                env={**os.environ, "LASER": laser_dir},
            )
            X = np.fromfile(str(output_path), dtype=np.float32, count=-1)
            X.resize(X.shape[0] // 1024, 1024)
        return X


class TfIdfEmbedding:
    def __init__(self, max_features: int = 15000, n_components: int = 5000) -> None:
        self.model = TfidfVectorizer(
            stop_words="english", binary=True, max_features=max_features
        )
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.fit_transform(cudf.Series(texts)).toarray().get()
        if self.pca.n_components < min(len(embeddings), self.model.max_features):
            return self.pca.fit_transform(embeddings)
        else:
            return embeddings

    def fit(self, texts: Union[List[str], pd.Series]) -> None:
        if self.model.max_features <= self.pca.n_components:
            self.model.fit(cudf.Series(texts))
        else:
            _embeddings = self.model.fit_transform(cudf.Series(texts)).toarray().get()
            self.pca.fit(_embeddings)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        model_filename = f"{path.stem}-max_feature={self.model.max_features}-n_component={self.pca.n_components}{path.suffix}"
        path = path.parent / model_filename
        dump({"model": self.model, "pca": self.pca}, str(path))
        return path

    @property
    def num_features(self) -> int:
        return min(self.model.max_features, self.pca.n_components)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TfIdfEmbedding":
        models = load(str(path))
        other = super().__new__(cls)
        other.model = models["model"]
        other.pca = models["pca"]
        return other

    def __call__(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        embeddings = self.model.transform(cudf.Series(texts)).toarray().get()
        if self.pca.n_components < self.model.max_features:
            return self.pca.transform(embeddings)
        else:
            return embeddings


def find_matches(
    posting_ids: List[str], embeddings: np.ndarray, threshold: float = 2.7
) -> List[List[str]]:
    KNN = min(max(3, len(posting_ids)), 50)

    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    predictions: List[List[str]] = []
    for k in range(embeddings.shape[0]):
        idx = np.where(
            distances[
                k,
            ]
            < threshold
        )[0]
        ids = indices[k, idx]
        predictions.append([posting_ids[int(i)] for i in ids])

    return predictions
