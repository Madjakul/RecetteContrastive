# preprocess.py

import logging

from recette_contrastive.utils.argparsers import PreprocessArgparse
from recette_contrastive.utils.data import Preprocessing
from recette_contrastive.utils.logger import logging_config

logging_config()


if __name__ == "__main__":
    logging.info("--- Starting preprocessing ---")
    args = PreprocessArgparse.parse_known_args()
    preprocessing = Preprocessing(
        model_name="sat-12l",
        output_dir=args.processed_ds_path,
    )

    with open("data/1.txt", "r", encoding="utf-8") as f:
        documents = f.read().splitlines()

    preprocessing.run(documents)
    logging.info("--- Preprocessing completed ---")
