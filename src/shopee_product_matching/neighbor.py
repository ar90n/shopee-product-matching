from typing import Callable, Union, Any, List, Dict
from sklearn.preprocessing import normalize

import cupy
import numpy as np
import torch
from cuml.neighbors import NearestNeighbors
from shopee_product_matching.util import get_device
from cupy.core.dlpack import toDlpack
from torch.utils.dlpack import from_dlpack


def create_match(
    name: str, **kwargs: Dict[str, Any]
) -> Callable[[Union[np.ndarray, torch.Tensor]], List[List[int]]]:
    if name == "knn":
        return KnnMatch(**kwargs)
    elif name == "cosine":
        return CosineSimilarityMatch(**kwargs)

    raise ValueError(f"unknown match name was given:{name}")


class KnnMatch:
    def __init__(self, threshold: float = 2.7, metric: str = "cosine", **kwargs: Dict[str, Any]) -> None:
        self._threshold = threshold
        self._metric = metric

    def __call__(self, embeddings: Union[np.ndarray, torch.Tensor]) -> List[List[int]]:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        KNN = min(max(3, len(embeddings)), 50)

        model = NearestNeighbors(n_neighbors=KNN, metric = self._metric)
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
    def __init__(self, threshold=0.3, chunk: int = 4096, knn:int = 256, **kwargs) -> None:
        self._threshold = threshold
        self._chunk = chunk
        self._knn = knn

    def __call__(
        self, embeddings: Union[np.ndarray, cupy.core.core.ndarray, torch.Tensor]
    ) -> List[List[int]]:
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        elif isinstance(embeddings, cupy.core.core.ndarray):
            embeddings = from_dlpack(toDlpack(embeddings))
        embeddings = embeddings.to(get_device()).to(torch.float32)
        embeddings *= 1.0 / (torch.norm(embeddings, dim=1).reshape(-1, 1) + 1e-12)

        CTS = len(embeddings) // self._chunk
        if (len(embeddings) % self._chunk) != 0:
            CTS += 1

        matches: List[List[int]] = []
        for j in range(CTS):
            a = j * self._chunk
            b = (j + 1) * self._chunk
            b = min(b, len(embeddings))
            print(f"{a} to {b}")
            distances = 1.0 - torch.matmul(embeddings, embeddings[a:b].T).cpu().T
            for k in range(b - a):
                ids = torch.where(
                    distances[
                        k,
                    ]
                    < self._threshold
                )[0][:self._knn]
                ids = list(ids)
                if len(ids) == 0:
                    ids.append(k + a)
                matches.append(ids)
        return matches