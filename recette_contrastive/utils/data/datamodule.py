# recette_contrastive/utils/data/datamodule.py

import logging
import os
import os.path as osp
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader

from recette_contrastive.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from recette_contrastive.utils.configs.base_config import BaseConfig


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        cfg: "BaseConfig",
        processed_ds_path: str,
        num_proc: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_proc = num_proc
        self.processed_ds_path = processed_ds_path
        self.cache_dir = cache_dir
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    def tokenize(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokenized_q = self.tokenizer(
            batch["anchor"],
            truncation=True,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
        )
        tokenized_pos = self.tokenizer(
            batch["positive"],
            truncation=True,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
        )
        return {
            "input_ids": tokenized_q["input_ids"],
            "attention_mask": tokenized_q["attention_mask"],
            "pos_input_ids": tokenized_pos["input_ids"],
            "pos_attention_mask": tokenized_pos["attention_mask"],
        }

    def setup(self, stage: Optional[str] = None) -> None:
        train_path = osp.join(self.processed_ds_path, "train")
        val_path = osp.join(self.processed_ds_path, "val")

        if osp.exists(train_path) and osp.exists(val_path):
            logging.info(f"Loading processed data from disk: {self.processed_ds_path}")
            self.train_ds = datasets.load_from_disk(train_path)
            self.val_ds = datasets.load_from_disk(val_path)
            return

        logging.info("Processed data not found. Running full preprocessing pipeline...")
        ds = datasets.load_dataset(
            "json",
            data_files="data/doc_*.jsonl",
            split="train",
        )
        columns = ds.column_names  # type: ignore
        ds = ds.train_test_split(test_size=0.1, shuffle=True)

        self.train_ds = ds["train"].map(  # type: ignore
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
            load_from_cache_file=self.cfg.data.load_from_cache_file,
        )
        self.val_ds = ds["test"].map(  # type: ignore
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
            load_from_cache_file=self.cfg.data.load_from_cache_file,
        )

        self.train_ds.set_format("torch")
        self.val_ds.set_format("torch")

        # --- Save the processed data to disk for future runs ---
        logging.info(f"Saving processed data to disk: {self.processed_ds_path}")
        os.makedirs(self.processed_ds_path, exist_ok=True)
        self.train_ds.save_to_disk(train_path)
        self.val_ds.save_to_disk(val_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            shuffle=self.cfg.data.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
        )
