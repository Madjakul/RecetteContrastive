# recette_contrastive/modules/modeling_contrastive_encoder.py

import logging
from typing import TYPE_CHECKING, Any, Dict

import lightning as L
import torch
from jaxtyping import Float, Int
from torcheval.metrics import HitRate, ReciprocalRank
from transformers import get_constant_schedule_with_warmup

from recette_contrastive.modules.info_nce_loss import InfoNCELoss
from recette_contrastive.modules.language_model import LanguageModel
from recette_contrastive.utils.helpers import flatten_dict

if TYPE_CHECKING:
    from recette_contrastive.utils.configs import BaseConfig


class ContrastiveEncoder(L.LightningModule):
    val_hr1: HitRate
    val_hr5: HitRate
    val_hr10: HitRate
    val_rr: ReciprocalRank

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        flat_params = flatten_dict(cfg.to_dict())
        self.save_hyperparameters(flat_params)
        self.cfg = cfg
        self.lm = LanguageModel(cfg)
        self.contrastive_loss = InfoNCELoss(cfg)
        self.fc1 = torch.nn.Linear(self.lm.hidden_size, self.lm.hidden_size)

    def configure_optimizers(self) -> Dict[str, Any]:
        logging.info(
            f"""Configuring optimizer: AdamW with lr={self.cfg.train.lr},
             weight_decay={self.cfg.train.weight_decay}."""
        )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        out = self.fc1(out)
        return out

    def training_step(self, batch, batch_idx: int) -> Float[torch.Tensor, ""]:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        q_mask = batch["attention_mask"]
        batch_size = q_embs.size(0)

        if self.trainer.world_size > 1 and self.cfg.train.gather:
            # all_gather adds a dimension at the start, so we flatten it with the batch dim
            # Shape changes from [num_gpus, batch_size, seq, hidden] -> [global_batch_size, seq, hidden]
            targets = (
                torch.arange(batch_size, device=q_embs.device)
                + batch_size * self.trainer.global_rank
            )
            all_pos_embs = self.all_gather(pos_embs, sync_grads=True).flatten(0, 1)
            all_pos_mask = self.all_gather(batch["pos_attention_mask"]).flatten(0, 1)

            loss_metrics = self.contrastive_loss(
                query_embs=q_embs,
                key_embs=all_pos_embs,
                q_mask=q_mask,
                k_mask=all_pos_mask,
                targets=targets,
            )
        else:
            targets = torch.arange(batch_size, device=q_embs.device)
            loss_metrics = self.contrastive_loss(
                query_embs=q_embs,
                key_embs=pos_embs,
                q_mask=q_mask,
                k_mask=batch["pos_attention_mask"],
                targets=targets,
            )

        loss = loss_metrics["loss"]

        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )
        return loss

    def on_validation_start(self):
        self.val_hr1 = HitRate(k=1, device=self.device)
        self.val_hr5 = HitRate(k=5, device=self.device)
        self.val_hr10 = HitRate(k=10, device=self.device)
        self.val_rr = ReciprocalRank(device=self.device)

    def validation_step(self, batch, batch_idx: int) -> None:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        q_mask = batch["attention_mask"]
        batch_size = q_embs.size(0)

        targets = torch.arange(batch_size, device=q_embs.device)
        loss_metrics = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=pos_embs,
            q_mask=q_mask,
            k_mask=batch["pos_attention_mask"],
            targets=targets,
        )

        all_scores = loss_metrics["all_scores"]
        batch_size = targets.size(0)
        self.val_hr1.update(all_scores, targets)
        self.val_hr5.update(all_scores, targets)
        self.val_hr10.update(all_scores, targets)
        self.val_rr.update(all_scores, targets)

    def on_validation_epoch_end(self) -> None:
        avg_hr1 = self.val_hr1.compute().mean()
        avg_hr5 = self.val_hr5.compute().mean()
        avg_hr10 = self.val_hr10.compute().mean()
        mrr = self.val_rr.compute().mean()
        self.log_dict(
            {
                "val_hr1": avg_hr1,
                "val_hr5": avg_hr5,
                "val_hr10": avg_hr10,
                "val_mrr": mrr,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_hr1.reset()
        self.val_hr5.reset()
        self.val_hr10.reset()
        self.val_rr.reset()
