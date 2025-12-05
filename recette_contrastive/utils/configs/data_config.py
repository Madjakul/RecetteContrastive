# recette_contrastive/utils/configs/data_config.py

from dataclasses import dataclass
from typing import Union

from recette_contrastive.utils.helpers import DictAccessMixin


@dataclass
class DataConfig(DictAccessMixin):
    batch_size: int = 32
    tokenizer_name: str = "almanach/camembertav2-base"
    max_length: int = 512
    padding: Union[bool, str] = False  # max_length
    map_batch_size: int = 1000
    load_from_cache_file: bool = False
    shuffle: bool = False
