import os
from pathlib import Path
from typing import Any, Optional
import wandb

from .util import get_project


def init_wandb_if_need() -> None:
    if wandb.run is None:
        wandb.init(project=get_project())


def save(path: str) -> None:
    init_wandb_if_need()

    wandb.save(path)


def restore(path: str) -> Any:
    init_wandb_if_need()

    return wandb.restore(path)