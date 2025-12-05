# recette_contrastive/utils/argparsers/train_argparse.py

import argparse


class TrainArgparse:

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(
            description="Arguments utilisés pour fine-tune un modèle."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Emplacement du fichier de configuration `train.yml`.",
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Dossier où vous souhaiter conserver vos logs d'entraînement.",
        )
        parser.add_argument(
            "--processed_ds_path",
            type=str,
            required=True,
            help="Emplacement des données d'entraînement déjà tokenisées.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=1,
            help=(
                "Nombre de processeurs à utiliser pour le multiprocessing. Le"
                " sampling de données pour l'entraînement peut s'effectuer en parallèle"
                " afin que le modèle reçoive les données plus vite. Le gain de vitesse"
                " est marginal si les données ont déjà été traîtées et tokenisées en"
                " avance. Plus de 3 processeurs devient presque inutile."
            ),
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="Dossier où les différents checkpoints du modèle fine-tune sera sauvegardé.",
        )
        args, _ = parser.parse_known_args()
        return args
