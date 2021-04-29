from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from shopee_product_matching import storage
from shopee_product_matching.logger import get_logger


def _get_ckpt_filename_fmt(
    config: Any, n_fold: Optional[int], ckpt_filename_base: Optional[str]
) -> str:
    ckpt_filename_fmt = "{epoch}-{val_loss:.2f}"
    if config.get("fold") is not None:
        fold_fmt = f"fold={config.fold}"
        if n_fold is not None:
            fold_fmt = f"{fold_fmt}@{n_fold - 1}"
        ckpt_filename_fmt = f"{fold_fmt}-{ckpt_filename_fmt}"
    if ckpt_filename_base is not None:
        ckpt_filename_fmt = f"{ckpt_filename_base}-{ckpt_filename_fmt}"
    return ckpt_filename_fmt


class ShopeeTrainer(pl.Trainer):
    def __init__(
        self,
        config: Any,
        precision: int = 16,
        gpus: int = 1,
        monitor: str = "valid_loss",
        mode: str = "min",
        ckpt_filename_base: Optional[str] = None,
    ) -> None:
        n_fold = config.get("n_fold")
        ckpt_filename_fmt = _get_ckpt_filename_fmt(config, n_fold, ckpt_filename_base)

        self._checkpoint_callback = ModelCheckpoint(
            verbose=True,
            dirpath="checkpoints",
            filename=ckpt_filename_fmt,
            monitor=monitor,
            mode=mode,
        )
        early_stopping_callback = EarlyStopping(
            patience=config.early_stop_patience,
            verbose=True,
            monitor=monitor,
            mode=mode,
        )
        callbacks = [self._checkpoint_callback, early_stopping_callback]

        super().__init__(
            precision=precision,
            gpus=gpus,
            callbacks=callbacks,
            max_epochs=config.max_epochs,
            overfit_batches=config.overfit_batches,
            fast_dev_run=config.fast_dev_run,
            logger=get_logger(),
        )

    def save_best_model(self) -> None:
        storage.save(self.best_model_path)

    @property
    def best_model_path(self) -> str:
        return self._checkpoint_callback.best_model_path
