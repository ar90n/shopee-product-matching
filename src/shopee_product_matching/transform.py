from os import error
from typing import Any, Callable, Dict, Iterable, Optional, cast

import pandas as pd
import torch
from shopee_product_matching.constants import Paths
from torchvision.io import read_image
from torchvision.transforms import Compose, Normalize, Resize

ImageTransform = Callable[[str], torch.Tensor]
TitleTransform = Callable[[str], torch.Tensor]


def identity(x: Any) -> Any:
    return x


def label_group_encoding_old(config: Any) -> Callable[[int], torch.Tensor]:
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



def label_group_encoding(config: Any) -> Callable[[int], torch.Tensor]:
    train_df = pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0)
    fold_df = pd.read_csv(Paths.requirements / "fold.csv", index_col=0)
    all_df = pd.concat(
        [train_df["label_group"], fold_df["fold"]],
        axis=1,
    )

    conv_dict: Dict[int, int] = {
        k: v
        for v, k in enumerate(
            all_df[all_df["fold"] != config.fold]["label_group"].unique()
        )
    }

    def _f(label_group: int) -> torch.Tensor:
        return torch.tensor(conv_dict[label_group])

    return _f


def imread(
    data_type: Optional[str] = None, is_cv: Optional[bool] = None
) -> Callable[[Any], Any]:
    if data_type is None and is_cv is None:
        raise ValueError("data_type or is_cv must be given")

    if data_type is None:
        data_type = "train_images" if is_cv else "test_images"

    def _f(image: Any) -> Any:
        image_path = Paths.shopee_product_matching / str(data_type) / str(image)
        return read_image(str(image_path)) / 255.0

    return _f


def read_resize_normalize(
    config: Any, data_type: Optional[str] = None
) -> Callable[[Any], Any]:
    return Compose(
        [
            imread(data_type, config.get("is_cv")),
            Resize(size=(config.image_size, config.image_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def map_stack(
    funcs: Iterable[Callable[[Any], torch.Tensor]]
) -> Callable[[Any], torch.Tensor]:
    def _f(input: Any) -> torch.Tensor:
        return torch.stack([f(input) for f in funcs])

    return _f