from pathlib import Path

from shopee_product_matching.util import get_input


class _TrainData:
    @property
    def posting_id_unique_count() -> int:
        return 34250

    @property
    def image_count() -> int:
        return 32412

    @property
    def image_phash_unique_count() -> int:
        return 28735

    @property
    def title_unique_unique_count() -> int:
        return 33117

    @property
    def label_group_unique_unique_count() -> int:
        return 11014


class _Paths:
    @property
    def shopee_product_matching(self) -> Path:
        return get_input() / "shopee-product-matching"


Paths = _Paths()
TrainData = _TrainData()