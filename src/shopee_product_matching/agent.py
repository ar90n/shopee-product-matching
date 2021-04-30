import os
import sys
from typing import Any, Optional

from shopee_product_matching.util import get_project

import wandb


def run(function: Any) -> None:
    if 1 < len(sys.argv):
        sweep_id = sys.argv[1]
    elif "SWEEP_ID" in os.environ:
        sweep_id = os.environ["SWEEP_ID"]
    else:
        raise EnvironmentError("missing sweep_id.")

    sweep_count: Optional[int] = 1
    if 2 < len(sys.argv):
        sweep_count = int(sys.argv[2])
    elif "SWEEP_COUNT" in os.environ:
        sweep_count = int(os.environ["SWEEP_COUNT"])
    if sweep_count is not None and sweep_count < 0:
        sweep_count = None

    wandb.agent(sweep_id, function=function, project=get_project(), count=sweep_count)
