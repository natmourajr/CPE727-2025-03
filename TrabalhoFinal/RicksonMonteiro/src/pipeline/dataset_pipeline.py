from __future__ import annotations
from pathlib import Path
import json

from src.core.config import ConfigLoader
from src.preprocess.dataset_builder import DatasetBuilder
from src.preprocess.canonical.dataset_saver import DatasetSaver
from src.preprocess.harmonizer import CategoryHarmonizer
from src.preprocess.transformer.goup_label_transform import GroupLabelTransformer
from src.ingestion.zip_loader import ZipLoader
from src.preprocess.parser import Parser 

class DatasetPipeline:
    """
    High-level pipeline to orchestrate dataset generation.
    This is the class executed by dataset.py
    """

    def __init__(self, config_path: Path):
        self.config_loader = ConfigLoader()

        # Load all configs
        self.dataset_cfg = self.config_loader.load_dataset()
        self.normalization_cfg = self.config_loader.load_normalization()
        self.canonical_cfg = self.config_loader.load_canonical().get("canonical", {})
        self.splitting_cfg = self.config_loader.load_splitting()
        self.validation_cfg = self.config_loader.load_validation()

        # merged config passed to DatasetBuilder
        self.full_cfg = {
            "normalization": self.normalization_cfg,
            "splitting": self.splitting_cfg,
            "validation": self.validation_cfg,
            "info": self.dataset_cfg["info"],
            "licenses": self.dataset_cfg["licenses"]
        }

        self.source_cfg = self.dataset_cfg["source"]

    # -----------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------
    def run(self):
        print("\nStarting DatasetPipeline...\n")

        experiments = self._load_experiments()
        parsed = self._parse_experiments(experiments)
        harmonized = self._harmonize(parsed)
        canonical = self._build_canonical(harmonized)

        saved_paths = self._save_canonical(canonical)
        # self._register_version(saved_paths, stats=getattr(canonical, "stats", None))

        grouped = self._group_transform(canonical)
        grouped_paths = self._save_grouped(grouped)
        # self._register_grouped_version(grouped_paths, grouped.get("statistics"))

        print("\nDatasetPipeline completed successfully!\n")

    # -----------------------------------------------------------
    # Modular steps
    # -----------------------------------------------------------
    def _load_experiments(self):
        loader = ZipLoader(
            zip_path=Path(self.source_cfg.get("path", "data/raw/yolo_files.zip")),
            extract_to=Path(self.source_cfg.get("extract_to", "data/interim/ingested/")),
            overwrite=self.source_cfg.get("overwrite", True),
            multi_experiment=self.source_cfg.get("multi_experiment", True)
        )
        experiments = list(loader.load())

        if not experiments:
            raise RuntimeError("No experiments available!")

        print(f"Loaded {len(experiments)} experiments")
        return experiments

    # def _detect_format(self, experiments):
    #     formats = {FormatDetector.detect(exp.root) for exp in experiments}
    #     if len(formats) != 1:
    #         raise ValueError(f"❌ Multiple annotation formats detected: {formats}")
    #     fmt = formats.pop()
    #     print(f"✔ Detected annotation format: {fmt}")
    #     return fmt

    def _parse_experiments(self, experiments):
        parsed = []
        print("Parsing experiments")
        for exp in experiments:
            parser = Parser(exp.root, experiment_id=exp.experiment_id_clean)
            parsed.append(parser.parse())
        print(f"\nParsed {len(parsed)} experiments.")
        return parsed

    def _harmonize(self, parsed):
        print("Harmonizing categories...")
        harmonized = CategoryHarmonizer.unify(parsed)
        return harmonized

    def _build_canonical(self, harmonized):
        print("Building canonical dataset...")
        builder = DatasetBuilder(
            root=Path("."),
            strict=False,
            config=self.full_cfg,
        )
        canonical = builder.build(harmonized)
        print("Canonical dataset built.")
        return canonical

    def _save_canonical(self, canonical):
        output_dir = Path(self.canonical_cfg["output_dir"])
        file_name = Path(self.canonical_cfg["filename"]).stem
        saver = DatasetSaver(root=output_dir)

        saved_paths = saver.save(
            canonical,
            name=file_name,
            formats=[self.canonical_cfg.get("file_format", "json")],
            overwrite=True
        )

        print("\nCanonical dataset saved:")
        for fmt, path in saved_paths.items():
            print(f"-> {fmt}: {path}")

        return saved_paths

    # def _register_version(self, paths, stats):
    #     registry = DatasetRegistry(root=Path(self.canonical_cfg["registry_dir"]))
    #     entry = registry.register(
    #         canonical_path=paths["json"],
    #         config=self.full_cfg,
    #         stats=stats
    #     )
    #     print("\n📌 Canonical version registered.")
    #     return entry

    def _group_transform(self, canonical):
        print("Applying GroupLabelTransformer...")
        transformer = GroupLabelTransformer(
            group_map=self.normalization_cfg.get("group_map")
        )
        grouped = transformer.apply(canonical).to_dict()
        print("Group transform complete.")
        return grouped

    def _save_grouped(self, grouped):
        output_dir = Path(self.canonical_cfg["output_dir"])
        saver = DatasetSaver(root=output_dir)

        name = Path(self.canonical_cfg["filename"]).stem + "_grouped"

        grouped_paths = saver.save(
            grouped,
            name=name,
            formats=[self.canonical_cfg.get("file_format", "json")],
            overwrite=True
        )

        print("\nGrouped dataset saved:")
        for fmt, path in grouped_paths.items():
            print(f" -> {fmt}: {path}")

        return grouped_paths

    # def _register_grouped_version(self, paths, stats):
    #     registry = DatasetRegistry(root=Path(self.canonical_cfg["registry_dir"]))
    #     registry.register(
    #         canonical_path=paths["json"],
    #         config=self.normalization_cfg,
    #         stats=stats
    #     )
    #     print("\n📌 Grouped version registered.")
