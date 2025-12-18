"""
validate.py
Entry point para executar a validação de um experimento específico.
"""

import argparse
from pathlib import Path

from src.pipeline.validate_pipeline import ValidatorPipeline
from src.trainer.utils import set_seed_everything


def main():
    parser = argparse.ArgumentParser(description="Unified validation runner")

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment folder"
    )

    args = parser.parse_args()
    experiment_path = Path(args.experiment)

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment folder not found: {experiment_path}")

    print(f"\nIniciando validação do experimento: {experiment_path}")

    orchestrator = ValidatorPipeline(experiment_path)
    orchestrator.run()
    orchestrator.save_report()

if __name__ == "__main__":
    set_seed_everything(42)
    main()
