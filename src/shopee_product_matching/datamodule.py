from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Union, cast

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .constants import Paths


class ShopeeProp(Enum):
    posting_id = "posting_id"
    image = "image"
    title = "title"
    image_phash = "image_phash"
    label_group = "label_group"

    def __str__(self) -> str:
        return str(self.value)


ShopeeRecord = NewType("ShopeeRecord", Dict[ShopeeProp, Union[Tensor, str]])


@dataclass
class ShopeeQuery:
    posting_id: bool = True
    image: Union[bool, Callable[[np.ndarray], np.ndarray]] = False
    title: bool = False
    image_phash: bool = False
    label_group: bool = False


class DatasetType(Enum):
    train: str = "train"
    test: str = "test"

    def __str__(self) -> str:
        return self.value


class ShopeeDataset(Dataset[ShopeeRecord]):
    df: pd.DataFrame
    query: ShopeeQuery
    dataset_type: DatasetType

    def __init__(self, df: pd.DataFrame, query: ShopeeQuery):
        self.df = df.reset_index()
        self.query = query
        self.dataset_type = self._detect_dataset_type(df)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> ShopeeRecord:
        row = self.df.iloc[index]
        image = cv2.imread(str(self._get_image_path(row.image)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        record = {}
        if self.query.posting_id:
            record[ShopeeProp.posting_id] = row.posting_id
        if isinstance(self.query.image, bool) and self.query.image:
            record[ShopeeProp.image] = torch.from_numpy(image.tranpose(2, 0, 1))
        if callable(self.query.image):
            augment_image = self.query.image(image)
            record[ShopeeProp.image] = torch.from_numpy(
                augment_image.transpose(2, 0, 1)
            )
        if self.query.title:
            record[ShopeeProp.title] = row.title
        if self.query.image_phash:
            record[ShopeeProp.image_phash] = row.image_phash
        if self.query.label_group:
            record[ShopeeProp.label_group] = torch.tensor(row.label_group)
        return cast(ShopeeRecord, record)

    def _detect_dataset_type(self, df: pd.DataFrame) -> DatasetType:
        if df.iloc[0].posting_id.startswith("train"):
            return DatasetType.train
        else:
            return DatasetType.test

    def _get_image_path(self, image: str) -> Path:
        if self.dataset_type == DatasetType.train:
            return Paths.shopee_product_matching / "train_images" / image
        else:
            return Paths.shopee_product_matching / "test_images" / image


@dataclass
class ShopeeDataModuleParam:
    train_query: ShopeeQuery
    valid_query: ShopeeQuery
    test_query: ShopeeQuery
    train_batch_size: int = 16
    valid_batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 4


class ShopeeDataModule(pl.LightningDataModule):
    param: ShopeeDataModuleParam

    def __init__(
        self,
        param: ShopeeDataModuleParam,
    ):
        self.param = param

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage=stage)
        train_full_df = pd.read_csv(Paths.shopee_product_matching / "train.csv")
        train_full_df["label_group"] = LabelEncoder().fit_transform(
            train_full_df["label_group"]
        )
        train_df, valid_df = train_test_split(
            train_full_df, test_size=0.2, random_state=42
        )
        self.train_dataset = ShopeeDataset(train_df, self.param.train_query)
        self.valid_dataset = ShopeeDataset(valid_df, self.param.valid_query)

        test_df = pd.read_csv(Paths.shopee_product_matching / "test.csv")
        self.test_dataset = ShopeeDataset(test_df, self.param.test_query)

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.param.train_batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self.param.num_workers,
        )

    def val_dataloader(
        self,
    ) -> Union[DataLoader[ShopeeRecord], List[DataLoader[ShopeeRecord]]]:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.param.valid_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.param.num_workers,
        )

    def test_dataloader(
        self,
    ) -> Union[DataLoader[ShopeeRecord], List[DataLoader[ShopeeRecord]]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.param.test_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.param.test_batch_size,
        )
