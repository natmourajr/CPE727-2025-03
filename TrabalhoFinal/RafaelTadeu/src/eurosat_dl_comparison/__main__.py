"""Application entry point."""
from pathlib import Path

from kedro.framework.project import configure_project


def main() -> None:
    """Entry point for running a Kedro project packaged with the package."""
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    from kedro.framework.cli import main as kedro_main  # noqa: PLC0415

    kedro_main()


if __name__ == "__main__":
    main()
