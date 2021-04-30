import gc
import os
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Compose
from typing_extensions import final
from shopee_product_matching.constants import Paths, seed

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


@contextmanager
def ensemble() -> None:

    cur_dir = str(Path.cwd().absolute())
    submissions = []
    with TemporaryDirectory(dir=cur_dir) as temp:
        try:
            os.chdir(temp)
            yield

            for p in Path.cwd(temp).glob("submission*.csv"):
                df = pd.read_csv(p, index_col=0)
                submissions.append(df)
        finally:
            os.chdir(cur_dir)
    print(submissions)


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
    )  # Decode original encoding


def get_model_path(model_name: str) -> Path:
    if is_kaggle():
        model_name = "".join(c for c in model_name if c != "=")
    return Paths.requirements / model_name
