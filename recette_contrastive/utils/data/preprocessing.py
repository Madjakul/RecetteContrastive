# recette_contrastive/utils/data/preprocessing.py

import json
import os.path as osp
import random
from typing import List, Tuple

import torch
from tqdm import tqdm
from wtpsplit import SaT


class Preprocessing:
    def __init__(
        self,
        model_name="sat-12l",
        output_dir="split_outputs",
        n_sentences=1,
    ):
        self.output_dir = output_dir
        self.n_sentences = n_sentences
        self.num_gpus = torch.cuda.device_count()
        self.model = SaT(model_name)
        self.model.half().to("cuda")

    def run(self, documents: List[str]):
        """
        documents: list of text strings
        """
        sampling_method = [self.sample_independent, self.sample_ict]

        for idx, document in enumerate(tqdm(documents, desc="Splitting documents")):
            sentences = self.model.split(document)
            with open(
                osp.join(self.output_dir, f"doc_{idx:05d}.jsonl"), "w", encoding="utf-8"
            ) as f_out:
                query, positive = random.choice(sampling_method)(
                    sentences, self.n_sentences
                )
                record = {
                    "doc_id": f"doc_{idx:05d}",
                    "query": query,
                    "positive": positive,
                }
                json.dump(record, f_out)
                f_out.write("\n")

    @staticmethod
    def sample_independent(sentences: List[str], n_sentences: int) -> Tuple[str, str]:
        min_required_sentences = 2 * n_sentences
        if len(sentences) < min_required_sentences:
            print("Not enough sentences for independent sampling.")
            return ("", "")

        query_start_idx = random.randint(0, len(sentences) - n_sentences)
        query = " ".join(sentences[query_start_idx : query_start_idx + n_sentences])

        if random.random() < 0.1:
            potential_positive = [
                sentences[query_start_idx : query_start_idx + 2 * n_sentences],
                sentences[
                    max(0, query_start_idx - n_sentences) : query_start_idx
                    + n_sentences
                ],
            ]
            positive_sentences = random.choice(potential_positive)
        else:
            positive_start_idx = random.randint(0, len(sentences) - n_sentences)
            positive_sentences = sentences[
                positive_start_idx : positive_start_idx + n_sentences
            ]

        positive = "".join(positive_sentences)
        return (query, positive)

    @staticmethod
    def sample_ict(sentences: List[str], n_sentences: int) -> Tuple[str, str]:
        context_size = 2 * n_sentences
        min_required_sentences = (3 * n_sentences) + context_size
        if len(sentences) < min_required_sentences:
            print("Not enough sentences for ICT sampling.")
            return ("", "")

        query_start_range = n_sentences
        query_end_range = len(sentences) - context_size
        if query_start_range >= query_end_range:
            return ("", "")

        query_idx = random.randint(query_start_range, query_end_range)
        anchor = " ".join(sentences[query_idx : query_idx + n_sentences])

        if random.random() < 0.1:
            positive_sentences = sentences[
                query_idx - n_sentences : query_idx + context_size
            ]
        else:
            positive_sentences = (
                sentences[query_idx - n_sentences : query_idx]
                + sentences[query_idx + n_sentences : query_idx + context_size]
            )
        positive = "".join(positive_sentences)

        return (anchor, positive)
