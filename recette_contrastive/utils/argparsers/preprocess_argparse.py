# recette_contrastive/utils/argparsers/preprocess_argparse.py

import argparse


class PreprocessArgparse:

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(
            description=(
                "Arguments utilisés pour créer des pairs de séquences à partir de PDFs."
            )
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Dossier où vous souhaiter conserver vos logs de preprocessing.",
        )
        parser.add_argument(
            "--processed_ds_path",
            type=str,
            required=True,
            help="Emplacement de vos données une fois traîtées et tokenisées.",
        )
        args, _ = parser.parse_known_args()
        return args
