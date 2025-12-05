# recette_contrastive/utils/configs/train_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from recette_contrastive.utils.helpers import DictAccessMixin


@dataclass
class TrainConfig(DictAccessMixin):
    tau: float = 0.07
    # --- optimizer ---
    lr: float = 1e-5
    weight_decay: float = 0.01
    # --- checkpointing ---
    checkpoint_metric: Literal["val_auroc", "val_mrr"] = "val_mrr"
    checkpoint_mode: Literal["min", "max"] = "max"
    save_top_k: int = 5
    # --- trainer ---
    gather: bool = True
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices: int = 1
    strategy: str = "ddp_find_unused_parameters_true"
    max_steps: int = -1
    max_epochs: int = 2
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: Optional[int] = None
    log_every_n_steps: int = 10
    accumulate_grad_batches: int = 8
    gradient_clip_val: Optional[float] = None
    precision: Literal["32", "16-mixed"] = "16-mixed"
    overfit_batches: float = 0.0
    # --- wandb ---
    use_wandb: bool = False
    log_model: bool = False
    watch: Literal["parameters", "gradients", "all", "none"] = "gradients"
