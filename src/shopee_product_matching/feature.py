import os
from pathlib import Path
from typing import Callable, Iterable, List, Union, Optional, Any, Tuple
from tempfile import TemporaryDirectory
import subprocess
import shutil

import cudf
import numpy as np
import pandas as pd
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from joblib import dump, load

# from sklearn.decomposition import PCA
from cuml import PCA
import fasttext


class LaserEmbedding:
    def __init__(
        self, lang: str, laser_dir: Optional[Path] = None, chunk_size: int = 4096
    ) -> None:
        self._lang = lang
        self._laser_dir = laser_dir
        self._chunk_size = chunk_size

    def _copy_laser_dir(self, dst: str) -> None:
        laser_dir = (
            self._laser_dir if self._laser_dir is not None else os.environ.get("LASER")
        )
        if laser_dir is None:
            raise EnvironmentError("Path to LASER directory is not given")

        shutil.copytree(str(laser_dir), f"{dst}/LASER")
        os.chmod(f"{dst}/LASER/tools-external/fastBPE/fast", 0o755)
        os.chmod(f"{dst}/LASER/tools-external/fastBPE/fast", 0o755)
        for p in Path(f"{dst}/LASER/tools-external/moses-tokenizer/tokenizer").glob(
            "*"
        ):
            os.chmod(str(p), 0o755)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        from more_itertools import chunked

        with TemporaryDirectory() as temp:
            self._copy_laser_dir(temp)
            laser_dir = Path(temp) / "LASER"
            embed_sh_path = laser_dir / "tasks" / "embed" / "embed.sh"

            results = []
            input_path = Path(temp) / "input.txt"
            output_path = Path(temp) / "output.raw"
            for chunk in chunked(texts, self._chunk_size):
                input_path.write_text("\n".join(chunk))
                subprocess.check_call(
                    args=[
                        "bash",
                        str(embed_sh_path),
                        str(input_path),
                        self._lang,
                        str(output_path),
                    ],
                    env={**os.environ, "LASER": str(laser_dir)},
                )
                X = np.fromfile(str(output_path), dtype=np.float32, count=-1)
                X.resize(X.shape[0] // 1024, 1024)
                output_path.unlink()
                results.append(X)
        X = np.vstack(results)
        return X


class TfIdfEmbedding:
    def __init__(self, max_features: int = 15000, n_components: int = 5000) -> None:
        self.model = TfidfVectorizer(
            stop_words="english", binary=True, max_features=max_features
        )
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.fit_transform(cudf.Series(texts)).toarray()
        return embeddings
        # if self.pca.n_components < min(len(embeddings), self.model.max_features):
        #    return self.pca.fit_transform(embeddings).get()
        # else:
        #    return embeddings.get()

    def fit(self, texts: Union[List[str], pd.Series]) -> None:
        if self.pca.n_components < min(len(texts), self.model.max_features):
            _embeddings = self.model.fit_transform(cudf.Series(texts)).toarray()
            self.pca.fit(_embeddings)
        else:
            self.model.fit(cudf.Series(texts))

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
    print("Finding similar titles...")
    CHUNK = 1024 * 4
    CTS = len(posting_ids) // CHUNK
    if (len(posting_ids) % CHUNK) != 0:
        CTS += 1

    import cupy

    preds = []
    for j in range(CTS):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(posting_ids))
        print("chunk", a, "to", b)

        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul(embeddings, embeddings[a:b].T).T
        for k in range(b - a):
            IDX = cupy.where(
                cts[
                    k,
                ]
                > 0.75
            )[0]
            o = [posting_ids[i] for i in cupy.asnumpy(IDX)]
            if posting_ids[k + a] not in o:
                o.append(posting_ids[k + a])
            preds.append(o)

    return preds


class FastTextEmbedding:
    _model: Any

    def __init__(
        self,
        word_ngrams: int = 4,
        epoch: int = 128,
        dim: int = 256,
        pretrained_vectors: str = "",
        model="skipgram",
        agg_func=None,
    ) -> None:
        self._model = None
        self._word_ngrams = word_ngrams
        self._epoch = epoch
        self._dim = dim
        self._pretrained_vectors = pretrained_vectors
        self._model = model
        self._agg_func = agg_func

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        with TemporaryDirectory() as temp:
            input_path = Path(temp) / "input.txt"
            input_path.write_text("\n".join(texts))
            self._model = fasttext.train_unsupervised(
                str(input_path),
                model=self._model,
                wordNgrams=self._word_ngrams,
                epoch=self._epoch,
                dim=self._dim,
                pretrainedVectors=self._pretrained_vectors,
            )

        result = []
        for text in texts:
            if self._agg_func is None:
                embedding = self._model.get_sentence_vector(text)
            else:
                word_embeddings = [(t, self._model[t]) for t in fasttext.tokenize(text)]
                embedding = self._agg_func(word_embeddings)
            result.append(embedding)
        return np.vstack(result)

    @classmethod
    def create_tfidf_agg_func(
        cls, tfidf_model: TfIdfEmbedding
    ) -> Callable[[List[Tuple[str, np.ndarray]]], np.ndarray]:
        word_to_index = {
            v: k
            for k, v in tfidf_model.model.get_feature_names()
            .to_pandas()
            .to_dict()
            .items()
        }
        idf = np.squeeze(tfidf_model.model.idf_)

        def _f(word_embeddings: List[Tuple[str, np.ndarray]]) -> np.ndarray:
            targets = [
                (word_to_index[w.lower()], v)
                for w, v in word_embeddings
                if w.lower() in word_to_index
            ]
            if len(targets) == 0:
                return np.mean(np.vstack([v for _, v in word_embeddings]), axis=0)

            word_idxs, embeddings = zip(
                *(
                    (word_to_index[w.lower()], vec)
                    for w, vec in word_embeddings
                    if w.lower() in word_to_index
                )
            )
            weights = idf[list(word_idxs)]
            weights /= np.sum(weights)
            return weights.get() @ np.vstack(embeddings)

        return _f