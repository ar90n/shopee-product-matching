from typing import Iterable, List

import numpy as np
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors
import cudf


class TfIdfEmbedding:
    def __init__(self, max_features: int = 15000, n_components: int = 5000) -> None:
        self.model = TfidfVectorizer(
            stop_words="english", binary=True, max_features=max_features
        )
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.fit_transform(cudf.Series(texts)).toarray()
        if self.pca.n_components < embeddings.shape[0]:
            embeddings = self.pca.fit_transform(embeddings).get()
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