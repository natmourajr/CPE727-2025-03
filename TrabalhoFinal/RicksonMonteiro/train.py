"""
train.py
Entry point para executar TrainingPipeline via linha de comando.
"""

import argparse
from pathlib import Path
from src.pipeline.training_pipeline import TrainingPipeline
from src.trainer.utils import set_seed_everything, load_yaml


def main():
    parser = argparse.ArgumentParser(description="Unified training runner")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path para arquivo YAML de configuração do treino"
    )

    args = parser.parse_args()

    config = Path(args.config)

    config = Path(args.config)
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    print(f"\nIniciando cross-validation com config: {config}")
    full_yaml = load_yaml(config)  # Apenas para validar o YAML
    print(full_yaml)
    for model, config_path in full_yaml.items():
        print(f" - Modelo configurado: {model}")
        print(f" - Configuração: {config_path}")
        pipeline = TrainingPipeline(config_path=config_path)
        pipeline.run()

if __name__ == "__main__":
    set_seed_everything(42)
    main()
