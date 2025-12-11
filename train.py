# train.py

import logging
import os

import torch

from recette_contrastive.modules import ContrastiveEncoder
from recette_contrastive.utils import train_utils
from recette_contrastive.utils.argparsers import TrainArgparse
from recette_contrastive.utils.configs import BaseConfig
from recette_contrastive.utils.logger import logging_config

logging_config()
torch.cuda.empty_cache()


if __name__ == "__main__":
    args = TrainArgparse.parse_known_args()
    cfg = BaseConfig().from_yaml(args.config_path)

    logging.info("--- Fine-tuning ---")
    model = ContrastiveEncoder(cfg)

    trainer = train_utils.setup_trainer(
        cfg=cfg,
        logs_dir=args.logs_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    logging.info(
        f"Process {os.getpid()} -- "
        f"Rank: {trainer.global_rank}, "
        f"World Size: {trainer.world_size}"
    )

    logging.info("Preparing data module...")
    num_proc = int(args.num_proc / max(1, trainer.world_size))
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_path=args.processed_ds_path,
        num_proc=num_proc,
        cache_dir=args.cache_dir,
    )
    dm.setup("fit")

    trainer.validate(model=model, datamodule=dm)
    torch.cuda.empty_cache()
    trainer.fit(model=model, datamodule=dm)
    logging.info("--- Fine-tuning finished ---")
