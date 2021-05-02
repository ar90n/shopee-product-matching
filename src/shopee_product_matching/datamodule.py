from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, cast

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .constants import Paths
from .util import string_escape


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
    posting_id: Optional[Callable[[Union[str, int, float]], torch.Tensor]] = None
    image: Optional[Callable[[Union[str, int, float]], torch.Tensor]] = None
    title: Optional[Callable[[Union[str, int, float]], torch.Tensor]] = None
    image_phash: Optional[Callable[[Union[str, int, float]], torch.Tensor]] = None
    label_group: Optional[Callable[[Union[str, int, float]], torch.Tensor]] = None


class ShopeeDataset(Dataset[ShopeeRecord]):
    df: pd.DataFrame
    query: ShopeeQuery

    def __init__(self, df: pd.DataFrame, query: ShopeeQuery):
        self.df = df.reset_index()
        self.query = query

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> ShopeeRecord:
        row = self.df.iloc[index]

        record = {}
        if self.query.posting_id is not None:
            record[ShopeeProp.posting_id] = self.query.posting_id(row.posting_id)
        if self.query.image is not None:
            record[ShopeeProp.image] = self.query.image(row.image)
        if self.query.title is not None:
            record[ShopeeProp.title] = self.query.title(row.title)
        if self.query.image_phash is not None:
            record[ShopeeProp.image_phash] = self.query.image_phash(row.image_phash)
        if self.query.label_group is not None:
            record[ShopeeProp.label_group] = self.query.label_group(row.label_group)
        return cast(ShopeeRecord, record)


@dataclass
class ShopeeDataModuleQueries:
    train: Optional[ShopeeQuery] = None
    valid: Optional[ShopeeQuery] = None
    test: Optional[ShopeeQuery] = None


DatasetTransform = Callable[[pd.DataFrame], pd.DataFrame]
DatasetSplit = Callable[[pd.DataFrame, Any], Tuple[pd.DataFrame, pd.DataFrame]]


@dataclass
class ShopeeDataModuleTransforms:
    train: Optional[DatasetTransform] = None
    valid: Optional[DatasetTransform] = None
    test: Optional[DatasetTransform] = None


class ShopeeDataModule(pl.LightningDataModule):
    _config: Any
    _queries: ShopeeDataModuleQueries
    _train_valid_split: Optional[DatasetSplit]
    _transforms: ShopeeDataModuleTransforms
    _train_dataset: Optional[ShopeeDataset] = None
    _valid_dataset: Optional[ShopeeDataset] = None
    _test_dataset: Optional[ShopeeDataset] = None

    def __init__(
        self,
        config: Any,
        queries: ShopeeDataModuleQueries,
        train_valid_split: Optional[DatasetSplit] = None,
        transforms: ShopeeDataModuleTransforms = ShopeeDataModuleTransforms(),
    ):
        super().__init__()
        self._config = config
        self._queries = queries
        self._train_valid_split = train_valid_split
        self._transforms = transforms

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        super().setup(stage=stage)
        train_full_df = cast(
            pd.DataFrame,
            pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0),
        )
        if self._train_valid_split is not None:
            train_df, valid_df = self._train_valid_split(train_full_df, self._config)
        else:
            train_df = train_full_df
            valid_df = train_full_df
        self._train_dataset = self._setup_dataset(
            train_df,
            self._queries.train,
            self._transforms.train,
        )
        self._valid_dataset = self._setup_dataset(
            valid_df,
            self._queries.valid,
            self._transforms.valid,
        )

        #test_df = pd.read_csv(Paths.shopee_product_matching / "test.csv", index_col=0)
        test_df = pd.read_csv(Paths.shopee_product_matching / "train.csv", index_col=0)
        self._test_dataset = self._setup_dataset(
            test_df,
            self._queries.test,
            self._transforms.test,
        )

    def _setup_dataset(
        self,
        df: pd.DataFrame,
        query: Optional[ShopeeQuery],
        transform: Optional[DatasetTransform],
    ) -> Optional[ShopeeDataset]:
        if query is None:
            return None

        if transform is not None:
            df = transform(df)
        return ShopeeDataset(df, query)

    def train_dataloader(self) -> Any:
        if self._train_dataset is None:
            raise ValueError("train dataset is not initialized")

        return DataLoader(
            self._train_dataset,
            batch_size=self._config.train_batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self._config.num_workers,
        )

    def val_dataloader(
        self,
    ) -> Union[DataLoader[ShopeeRecord], List[DataLoader[ShopeeRecord]]]:
        if self._valid_dataset is None:
            raise ValueError("valid dataset is not initialized")

        return DataLoader(
            self._valid_dataset,
            batch_size=self._config.valid_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self._config.num_workers,
        )

    def test_dataloader(
        self,
    ) -> Union[DataLoader[ShopeeRecord], List[DataLoader[ShopeeRecord]]]:
        if self._test_dataset is None:
            raise ValueError("test dataset is not initialized")
        return DataLoader(
            self._test_dataset,
            batch_size=self._config.test_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self._config.num_workers,
        )
