from pathlib import Path


class _TrainData:
    @property
    def posting_id_unique_count(self) -> int:
        return 34250

    @property
    def image_count(self) -> int:
        return 32412

    @property
    def image_phash_unique_count(self) -> int:
        return 28735

    @property
    def title_unique_unique_count(self) -> int:
        return 33117

    @property
    def label_group_unique_unique_count(self) -> int:
        return 11014

    @property
    def label_group_unique_count_pdf_fold(self) -> int:
        return 8812


class _Paths:
    @property
    def shopee_product_matching(self) -> Path:
        from shopee_product_matching.util import get_input

        return get_input() / "shopee-product-matching"

    @property
    def requirements(self) -> Path:
        from shopee_product_matching.util import get_input

        return get_input() / "shopeeproductmatchingrequirements"

    @property
    def requirements2(self) -> Path:
        from shopee_product_matching.util import get_input

        return get_input() / "shopeeproductmatchingrequirements2"


Paths = _Paths()
TrainData = _TrainData()
seed = 42
