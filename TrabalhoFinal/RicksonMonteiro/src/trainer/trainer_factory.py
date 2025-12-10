"""
TrainerFactory
---------------

Factory responsável por retornar o Trainer correto com base no nome do modelo,
mantendo o pipeline desacoplado e escalável.

Uso:

    trainer_class = TrainerFactory.get("retinanet")
    trainer = trainer_class(...)
"""

from __future__ import annotations

from src.trainer.retinanet_trainer import RetinaNetTrainer
from src.trainer.frcnn_trainer import FasterRCNNTrainer
from src.trainer.ssd_trainer import SSDLiteTrainer


class TrainerFactory:
    REGISTRY = {
        "retinanet": RetinaNetTrainer,
        "fasterrcnn": FasterRCNNTrainer,
        "ssd": SSDLiteTrainer,
    }

    @classmethod
    def get(cls, model_name: str):
        """
        Retorna a classe Trainer correta baseada no nome do modelo.

        Args:
            model_name (str): ex: "retinanet", "fasterrcnn", "ssd"

        Returns:
            Trainer class

        Raises:
            KeyError se o nome não estiver registrado
        """

        key = model_name.lower()

        if key not in cls.REGISTRY:
            raise KeyError(
                f"[TrainerFactory] Modelo desconhecido: '{model_name}'. "
                f"Treinadores disponíveis: {list(cls.REGISTRY.keys())}"
            )

        return cls.REGISTRY[key]

    @classmethod
    def register(cls, model_name: str, trainer_class):
        """
        Permite adicionar novos trainers dinamicamente.

        Exemplo:
            TrainerFactory.register("yolov8", YOLOv8Trainer)
        """

        key = model_name.lower()

        if key in cls.REGISTRY:
            print(f"[TrainerFactory] Aviso: '{model_name}' já existia e será sobrescrito.")

        cls.REGISTRY[key] = trainer_class

        print(f"[TrainerFactory] Trainer '{model_name}' registrado com sucesso!")
