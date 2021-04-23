import atexit
import gc
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, List

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations import Compose
from wandb import wandb_agent

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
    cur_path = Path.cwd()
    while True:
        candidate = cur_path / "input"
        if candidate.exists():
            return candidate

        if cur_path == cur_path.parent:
            break
        cur_path = cur_path.parent

    raise EnvironmentError("input is not found")


def get_requirements() -> Path:
    return get_input() / "shopeeproductmatchingrequirements"


def get_device(n: Optional[int] = None) -> Any:
    if is_tpu_available():
        return xm.xla_device(n=n, devkind="TPU")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_project() -> str:
    return os.environ.get("WANDB_PROJECT", "shopee-product-matching")


def get_notebook_name() -> Optional[str]:
    return os.environ.get("NOTEBOOK_NAME")


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


def initialize(
    seed: int,
    job_type: JobType,
    params: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
) -> None:
    pl.seed_everything(seed)

    if has_wandb and job_type == JobType.Training:
        group = os.environ.get("EXPERIMENT_NAME")

        wandb.init(
            id=id,
            name=get_notebook_name(),
            group=group,
            job_type=str(job_type),
            project=get_project(),
            resume="allow",
            config=params,
        )

        commit_hash = os.environ.get("GIT_COMMIT_HASH")
        if commit_hash is not None:
            wandb.config.update({"commit_hash": commit_hash})

        memo = os.environ.get("MEMO")
        if memo is not None:
            wandb.config.update({"memo": memo})


def finalize() -> None:
    wandb.finish()


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