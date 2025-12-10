from pathlib import Path
import yaml


class ConfigLoader:
    """
    Centralized loader for all YAML configuration files under /configs/.
    Handles:
        - dataset, parsing, validation, normalization, canonical...
        - alias_map + group_map integration
        - experiment.yaml include system
    """

    def __init__(self, base_dir: Path = Path("config")):
        self.base_dir = base_dir

    # ---------------------------------------------------------
    # Internal YAML reader
    # ---------------------------------------------------------
    def _read_yaml(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"[ConfigLoader] File not found: {path}")

        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"[ConfigLoader] Invalid YAML file: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"[ConfigLoader] Failed to parse YAML: {path}\n{e}")

    def load(self, relative_path: str) -> dict:
        return self._read_yaml(self.base_dir / relative_path)

    # ---------------------------------------------------------
    # Specific module loaders
    # ---------------------------------------------------------

    def load_dataset(self) -> dict:
        return self.load("dataset.yaml")

    def load_parsing(self) -> dict:
        return self.load("parsing.yaml")

    def load_validation(self) -> dict:
        return self.load("validation.yaml")

    def load_canonical(self) -> dict:
        return self.load("canonical.yaml")

    def load_splitting(self) -> dict:
        return self.load("splitting.yaml")

    def load_statistics(self) -> dict:
        return self.load("statistics.yaml")

    # ---------------------------------------------------------
    # Normalization (merged config)
    # ---------------------------------------------------------
    def load_normalization(self) -> dict:
        base = self.load("normalization/normalization.yaml")

        alias_path = self.base_dir / "normalization" / "alias_map.yaml"
        if alias_path.exists():
            base["alias_map"] = self._read_yaml(alias_path).get("alias_map", {})

        group_path = self.base_dir / "normalization" / "group_map.yaml"
        if group_path.exists():
            base["group_map"] = self._read_yaml(group_path).get("group_map", {})

        return base

    # ---------------------------------------------------------
    # Experiment (aggregated config)
    # ---------------------------------------------------------
    def load_experiment(self) -> dict:
        exp = self.load("experiment.yaml")
        exp_cfg = exp.get("experiment", {})

        includes = exp_cfg.get("include", [])
        final = {"experiment": exp_cfg}

        for file in includes:
            # allow nested paths
            path = self.base_dir / file
            if not path.exists():
                raise FileNotFoundError(
                    f"[ConfigLoader] Included file not found: {file}"
                )

            # if including the whole normalization module
            if file.startswith("normalization"):
                final["normalization"] = self.load_normalization()
                continue

            resolved = self.load(file)
            name = Path(file).stem  # e.g. dataset.yaml → dataset
            final[name] = resolved

        return final

    # ---------------------------------------------------------
    # Tracking
    # ---------------------------------------------------------

    def load_tracking(self) -> dict:
        return {
            "mlflow": self.load("tracking/mlflow.yaml"),
            "dvc": self.load("tracking/dvc.yaml"),
        }

    def load_mlflow(self) -> dict:
        return self.load("tracking/mlflow.yaml")

    def load_dvc(self) -> dict:
        return self.load("tracking/dvc.yaml")
