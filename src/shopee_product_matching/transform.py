from typing import Any, Callable, Dict

import pandas as pd
import torch
from shopee_product_matching.constants import Paths
from torchvision.io import read_image
from torchvision.transforms import Compose, Normalize, Resize

ImageTransform = Callable[[str], torch.Tensor]


def identity(x: Any) -> Any:
    return x


def label_group_encoding(config: Any) -> Callable[[int], torch.Tensor]:
    train_df = pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0)
    fold_df = pd.read_csv(Paths.requirements / "fold.csv", index_col=0)
    all_df = pd.concat(
        [train_df["label_group"], fold_df["label_group"].rename("label_encoding")],
        axis=1,
    )
    conv_dict: Dict[int, int] = (
        all_df.drop_duplicates("label_group")
        .set_index("label_group")
        .to_dict("series")["label_encoding"]
    )

    def _f(label_group: int) -> torch.Tensor:
        return torch.tensor(conv_dict[label_group])

    return _f


def imread(data_type: str) -> Callable[[Any], Any]:
    def _f(image: Any) -> Any:
        image_path = Paths.shopee_product_matching / data_type / image
        return read_image(str(image_path)) / 255.0

    return _f


def read_resize_normalize(config: Any, data_type: str) -> Callable[[Any], Any]:
    return Compose(
        [
            imread(data_type),
            Resize(size=(config.image_size, config.image_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
