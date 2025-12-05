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
            "--config_path",
            type=str,
            required=True,
            help="Emplacement du fichier de configuration `preprocess.yml`.",
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
        parser.add_argument(
            "--num_proc",
            type=int,
            default=1,
            help=(
                "Nombre de processeurs à utiliser pour le multiprocessing. Le"
                " La transformation de PDFs en text ainsi que le sampling de passage"
                " peut être considérablement acceléré si effectué en parallèle."
                " Un grand nombre de processeurs est conseillé ici."
            ),
        )
        args, _ = parser.parse_known_args()
        return args
