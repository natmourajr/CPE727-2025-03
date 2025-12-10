"""
dataset.py
Entry point for running the DatasetPipeline from the command line.
"""

import argparse
from pathlib import Path

from src.pipeline.dataset_pipeline import DatasetPipeline


def main():
    parser = argparse.ArgumentParser(description="Dataset generation orchestrator")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML dataset configuration"
    )

    args = parser.parse_args()

    pipeline = DatasetPipeline(config_path=Path(args.config))
    pipeline.run()


if __name__ == "__main__":
    main()
