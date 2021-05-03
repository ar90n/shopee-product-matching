from typing import Callable, Union, Any, List, Dict
from sklearn.preprocessing import normalize

import cupy
import numpy as np
import torch
from cuml.neighbors import NearestNeighbors
from shopee_product_matching.util import get_device


def create_match(
    name: str, **kwargs: Dict[str, Any]
) -> Callable[[Union[np.ndarray, torch.Tensor]], List[List[int]]]:
    if name == "knn":
        return KnnMatch(**kwargs)
    elif name == "cosine":
        return CosineSimilarityMatch(**kwargs)

    raise ValueError(f"unknown match name was given:{name}")


class KnnMatch:
    def __init__(self, threshold: float = 2.7, **kwargs: Dict[str, Any]) -> None:
        self._threshold = threshold

    def __call__(self, embeddings: Union[np.ndarray, torch.Tensor]) -> List[List[int]]:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        KNN = min(max(3, len(embeddings)), 50)

        model = NearestNeighbors(n_neighbors=KNN)
        model.fit(embeddings)
        distances, indices = model.kneighbors(embeddings)

        matches: List[List[int]] = []
        for k in range(embeddings.shape[0]):
            idx = np.where(
                distances[
                    k,
                ]
                < self._threshold
            )[0]
            ids = indices[k, idx]
            matches.append([int(i) for i in ids])
        return matches


class CosineSimilarityMatch:
    def __init__(self, threshold=0.3, chunk: int = 4096, **kwargs) -> None:
        self._threshold = threshold
        self._chunk = chunk

    def __call__(self, embeddings: Union[np.ndarray, torch.Tensor]) -> List[List[int]]:
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        embeddings = embeddings.to(get_device())
        embeddings *= 1.0 / (torch.norm(embeddings, dim=1).reshape(-1, 1) + 1e-12)

        CTS = len(embeddings) // self._chunk
        if (len(embeddings) % self._chunk) != 0:
            CTS += 1

        matches: List[List[int]] = []
        for j in range(CTS):
            a = j * self._chunk
            b = (j + 1) * self._chunk
            b = min(b, len(embeddings))
            distances = 1.0 - torch.matmul(embeddings, embeddings[a:b].T).cpu().T
            for k in range(b - a):
                ids = (
                    torch.where(
                        distances[
                            k,
                        ]
                        < self._threshold
                    )[0]
                )
                matches.append([int(i) for i in ids])
        return matches