from __future__ import annotations
from pathlib import Path
import json
import shutil
from datetime import datetime
import yaml
import copy
from src.cross_validation.cross_validator import CrossValidator
from src.trainer.trainer_factory import TrainerFactory


def load_yaml(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class TrainingPipeline:
    """
    Orquestra o treinamento:
    1. gera folds (CrossValidator)
    2. treina cada fold com trainer específico
    3. salva métricas individuais
    4. seleciona melhor fold e salva best_model global
    """

    def __init__(self, config_path: Path):
        self.cfg = load_yaml(config_path)

        self.model_name = self.cfg["model"]
        self.model_cfg = self.cfg["model_cfg"]
        self.training_cfg = self.cfg["training"]

        self.num_folds = self.cfg["cross_validation"]["num_folds"]
        self.seed = self.cfg["seed"]

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.run_dir = Path(f"experiments/{self.model_name}/run_{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        print(f"\nExperimento criado em: {self.run_dir}")

        yaml_save_path = self.run_dir / "config.yaml"
        with open(yaml_save_path, "w") as f:
            yaml.dump(self.cfg, f)

        print(f"Arquivo de configuração salvo em: {yaml_save_path}")


    # -------------------------------------------------------------------
    def run(self):
        print("\nIniciando TrainingPipeline...")

        # Caminho para canonical.json
        canonical_json = self.cfg["dataset"]["canonical_json"]
        print(f"Dataset: {canonical_json}")

        validator = CrossValidator(
            canonical_json=canonical_json,
            run_dir=self.run_dir,
            num_folds=self.num_folds,
            seed=self.seed
        )

        fold_dirs = validator.run()


        TrainerClass = TrainerFactory.get(self.model_name)

        best_fold_metric = -1
        best_fold_model = None
        best_fold_name = None

        # -----------------------------------------------------------
        # 3️⃣ Treinar cada fold
        # -----------------------------------------------------------
        for fold_dir in fold_dirs:
            print(f"\n==============================")
            print(f"Treinando {self.model_name} — {fold_dir.name}")
            print(f"==============================")

            train_json = fold_dir / "train.json"
            val_json = fold_dir / "val.json"

            trainer = TrainerClass(
                training_cfg=copy.deepcopy(self.training_cfg),
                model_cfg=copy.deepcopy(self.model_cfg),
                fold_dir=fold_dir,
                train_json=train_json,
                val_json=val_json
            )

            fold_metric, fold_model_path = trainer.train()

            # salvar no experimento (opcional)
            shutil.copy(fold_model_path, self.models_dir / f"{fold_dir.name}_best.pth")

            # selecionar melhor fold
            if fold_metric > best_fold_metric:
                best_fold_metric = fold_metric
                best_fold_model = fold_model_path
                best_fold_name = fold_dir.name


        final_path = self.models_dir / "best_model.pth"
        shutil.copy(best_fold_model, final_path)

        print("\nMelhor fold:", best_fold_name)
        print("Melhor mAP:", best_fold_metric)
        print("Modelo final salvo em:", final_path)

        print("\nPipeline finalizada com sucesso!")
