# recette_contrastive/utils/configs/preprocess_config.py

from dataclasses import dataclass

from recette_contrastive.utils.helpers import DictAccessMixin


@dataclass
class PreprocessConfig(DictAccessMixin):
    n_sample_per_doc: int = 50
    independent_rate: float = 0.6
    exact_match_rate: float = 0.1
