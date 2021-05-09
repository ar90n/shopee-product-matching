import gc
import os
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Compose
from typing_extensions import final
from shopee_product_matching.constants import Paths, seed
from shopee_product_matching.metric import f1_score

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    has_xla = True
except ImportError:
    has_xla = False

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


class JobType(Enum):
    Training = "training"
    Inferene = "inference"

    def __str__(self) -> str:
        return self.value


class Runtime(Enum):
    Python = "python"
    IPython = "ipython"
    Notebook = "notebook"

    def __str__(self) -> str:
        return self.value


def runtime_type() -> Runtime:
    try:

        from IPython.core.getipython import get_ipython

        if "terminal" in get_ipython().__module__:
            return Runtime.IPython
        else:
            return Runtime.Notebook
    except (ImportError, NameError, AttributeError):
        return Runtime.Python


def is_tpu_available() -> bool:
    return has_xla and ("TPU_NAME" in os.environ)


def is_kaggle() -> bool:
    return "KAGGLE_URL_BASE" in os.environ


def is_notebook() -> bool:

    return runtime_type() == Runtime.Notebook


def get_input() -> Path:
    if is_kaggle():
        return Path("/kaggle/input")

    cur_path = Path.cwd()
    while True:
        candidate = cur_path / "input"
        if candidate.exists():
            return candidate

        if cur_path == cur_path.parent:
            break
        cur_path = cur_path.parent

    raise EnvironmentError("input is not found")


def get_device(n: Optional[int] = None) -> Any:
    if is_tpu_available():
        return xm.xla_device(n=n, devkind="TPU")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_project() -> str:
    return "shopee-product-matching"


def clean_up() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def to_device(tensors, device=None) -> Any:
    if device is None:
        device = get_device()

    if isinstance(tensors, Iterable):
        ret = [t.to(device) for t in tensors]
        if isinstance(tensors, tuple):
            ret = tuple(ret)
        return ret

    return tensors.to(device)


def exit() -> None:
    if is_kaggle() and is_notebook():
        _ = [0] * 64 * 1000000000
    else:
        sys.exit(1)


def _get_group() -> str:
    if "EXPERIMENT_NAME" in os.environ:
        return os.environ["EXPERIMENT_NAME"]
    else:
        return Path(sys.argv[0]).absolute().parent.stem


@contextmanager
def context(
    config_defaults: Dict[str, Any],
    job_type: JobType,
) -> None:
    pl.seed_everything(seed)

    mode = "online" if job_type == JobType.Training else "disabled"
    wandb.init(
        group=_get_group(),
        job_type=str(job_type),
        project=get_project(),
        config=config_defaults,
        mode=mode,
    )

    commit_hash = os.environ.get("GIT_COMMIT_HASH")
    if commit_hash is not None:
        wandb.config.update({"commit_hash": commit_hash})

    memo = os.environ.get("MEMO")
    if memo is not None:
        wandb.config.update({"memo": memo})

    try:
        yield wandb.config
    finally:
        if mode == "online":
            wandb.finish()

    if wandb.config.get("is_cv", False):
        submission = pd.read_csv("submission.csv", index_col=0)
        submission_matches = submission["matches"].map(lambda x: x.split(" "))
        exp_df = pd.read_csv(
            Paths.shopee_product_matching / "train.csv", index_col=0
        ).loc[submission.index]
        exp_matches = get_matches(
            exp_df.index,
            exp_df["label_group"],
        )

        f1 = f1_score(submission_matches, exp_matches)
        print(f"f1:{f1}")


@contextmanager
def ensemble(filename="submission.csv", conf_threshold=0.75) -> None:
    def merge_matches(submissions: List[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(submissions)
            .groupby("posting_id")
            .sum()
            .applymap(lambda x: " ".join(set(x)))
        )

    def merge_distances(submissions: List[pd.DataFrame], th: float) -> pd.DataFrame:
        ws = []
        for i, s in enumerate(submissions):
            s["i"] = i
            ws.append(s.iloc[0]["weight"])
        df = pd.concat(submissions)
        total_weights = df.groupby(["posting_id"])[["i"]].agg(
            lambda x: sum(ws[i] for i in set(x))
        )

        # dist was aslready multiplied by weight
        df = df.groupby(["posting_id", "neighbor"])[["dist"]].sum()
        df = df.reset_index()
        df["r"] = df["dist"] / df["posting_id"].map(total_weights.i)
        df["neighbor"] = df["neighbor"].apply(lambda x: [x])
        res = pd.DataFrame(
            df[th < df["r"]].groupby("posting_id")["neighbor"].sum()
        ).rename(columns={"neighbor": "matches"})
        return res

    def merge_embeddings(
        embeddings: List[Dict[str, Any]], submission: pd.DataFrame
    ) -> torch.Tensor:
        embedding_map: Dict[str, torch.Tensor] = {}
        for e in embeddings:
            cur = {k: v for k, v in zip(e["posting_id"], e["embedding"])}
            embedding_map = {**embedding_map, **cur}

        result = []
        for p in submission.index.values:
            result.append(embedding_map[p])
        return torch.vstack(result)

    cur_dir = str(Path.cwd().absolute())
    submissions = []
    embeddings = []
    distances = []
    with TemporaryDirectory(dir=cur_dir) as temp:
        try:
            os.chdir(temp)
            yield

            for p in Path(temp).glob("submission*.csv"):
                df = pd.read_csv(p, index_col=0).applymap(lambda x: x.split())
                submissions.append(df)

            for p in Path(temp).glob("submission*.pt"):
                embeddings.append(torch.load(p))

            for p in Path(temp).glob("submission*.pkl"):
                distances.append(pd.read_pickle(p))
        finally:
            os.chdir(cur_dir)

    if 0 < len(distances):
        dist_submission = merge_distances(distances, conf_threshold)
        submissions.append(dist_submission)

    submission = merge_matches(submissions)
    submission.to_csv(filename)

    if 0 < len(embeddings):
        embedding = merge_embeddings(embeddings, submission)
        torch.save(embedding, "submission.pt")


def pass_as_image(func: Compose) -> Callable[[np.ndarray], np.ndarray]:
    def _f(image: np.ndarray) -> np.ndarray:
        return func(image=image)["image"]

    return _f


# https://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python
def get_params_by_inspection() -> Dict[str, Any]:
    import inspect

    params = {}
    param_names = (
        os.environ["PARAM_NAMES"].split(",") if "PARAM_NAMES" in os.environ else []
    )

    frame = inspect.currentframe()
    try:
        parent_local_variables = frame.f_back.f_locals
        for name in param_names:
            params[name] = parent_local_variables[name]
    finally:
        del frame
    return params


def save_submission_csv(
    posting_ids: List[str], matches: List[List[str]], filename: Optional[str]
) -> None:
    df = pd.DataFrame(
        {
            "posting_id": posting_ids,
            "matches": [" ".join(rs) for rs in matches],
        }
    )

    filename = "submission.csv" if filename is None else filename
    df.to_csv(filename, index=False)


def save_submission_embedding(
    posting_ids: List[str],
    embeddings: Union[np.ndarray, torch.Tensor],
    filename: Optional[str],
) -> None:
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    filename = (
        "submission.pt" if filename is None else str(Path(filename).with_suffix(".pt"))
    )
    torch.save({"posting_id": posting_ids, "embedding": embeddings}, filename)


def save_submission_confidence(
    posting_ids: List[str],
    embeddings: Union[np.ndarray, torch.Tensor],
    threshold: float,
    filename: Optional[str],
    weight: float = 1.0,
) -> None:
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    chunk = 4096
    embeddings = embeddings.to(get_device()).to(torch.float32)
    embeddings *= 1.0 / (torch.norm(embeddings, dim=1).reshape(-1, 1) + 1e-12)

    CTS = len(embeddings) // chunk
    if (len(embeddings) % chunk) != 0:
        CTS += 1

    matches: Dict[str, List[int]] = {}
    for j in range(CTS):
        a = j * chunk
        b = (j + 1) * chunk
        b = min(b, len(embeddings))
        print(f"{a} to {b}")
        distances = torch.matmul(embeddings, embeddings[a:b].T).cpu().T
        distances = torch.clamp(distances, 0.0, 1.0)
        for k in range(b - a):
            ids = torch.where((1 - threshold) < distances[k,])[
                0
            ][:256]
            ids = list(ids)
            dists = [float(v) for v in distances[k, ids]]
            pids = [posting_ids[i] for i in ids]
            values = list(zip(pids, dists))
            matches[posting_ids[(k + a)]] = values

    records = []
    for k0, vs in matches.items():
        for k1, d in vs:
            records.append(
                {"posting_id": k0, "neighbor": k1, "dist": weight * d, "weight": weight}
            )
    df = pd.DataFrame.from_records(records)
    filename = (
        "submission.pkl"
        if filename is None
        else str(Path(filename).with_suffix(".pkl"))
    )
    df.to_pickle(filename)


def clean_up() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# https://www.kaggle.com/c/shopee-product-matching/discussion/233605
def string_escape(s, encoding="utf-8"):
    return (
        s.encode("latin1")  # To bytes, required by 'unicode-escape'
        .decode("unicode-escape")  # Perform the actual octal-escaping decode
        .encode("latin1")  # 1:1 mapping back to bytes
        .decode(encoding)
        .replace("\n", "")
    )  # Decode original encoding


def get_model_path(model_name: str) -> Path:
    if is_kaggle():
        model_name = "".join(c for c in model_name if c != "=")

    p = Paths.requirements2 / model_name
    if p.exists():
        return p

    return Paths.requirements / model_name


def get_matches(posting_ids: List[str], label_groups: List[str]) -> List[List[str]]:
    df = pd.DataFrame(
        {
            "label_group": label_groups,
            "posting_id": posting_ids,
        }
    )
    return (
        df["label_group"]
        .map(df.groupby(["label_group"])["posting_id"].unique().to_dict())
        .values.tolist()
    )