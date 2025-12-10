# src/dataset/dataset_builder.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.preprocess.canonical.dataset_canonical import DatasetCanonical

from src.preprocess.validation.structure_validator import StructureValidator
from src.preprocess.validation.bbox_validator import BBoxValidator
from src.preprocess.validation.bbox_corrector import BBoxCorrector
from src.preprocess.validation.validation_report import ValidationReport

from src.preprocess.normalization.dataset_normalizer import DatasetNormalizer
from src.dataset.statistics.dataset_statistics import DatasetStatistics
# from src.dataset. import SplitStrategyFactory


class DatasetBuilder:
    """
    DatasetBuilder

    High-level orchestration for generating a **fully canonical dataset**
    from a harmonized multi-experiment dataset (output of CategoryHarmonizer).

    Responsibilities:
        ✔ Structural validation
        ✔ Bounding-box validation & correction
        ✔ Category normalization (aliases, snake_case, ordering)
        ✔ Statistics computation
        ✔ Train/val/test split creation
        ✔ Final canonical packaging (DatasetCanonical)

    NOTE:
        This builder **no longer deals with groups**, since category groups
        are now handled by dedicated transformers outside the canonical pipeline.
    """

    def __init__(
        self,
        root: Path,
        strict: bool = True,
        classes: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.root = Path(root)
        self.strict = strict
        self.classes = classes
        self.config = config or {}

        # ------------------------------------------------------------
        # Validators
        # ------------------------------------------------------------
        self.structure_validator = StructureValidator(strict=strict)
        self.bbox_validator = BBoxValidator(strict=strict)
        self.bbox_corrector = BBoxCorrector()

        # ------------------------------------------------------------
        # Normalizer configuration
        # ------------------------------------------------------------
        norm_cfg = self.config.get("normalization", {})

        self.normalizer = DatasetNormalizer(
            classes=classes,
            alias_map=norm_cfg.get("alias_map"),
            lowercase_classes=norm_cfg.get("lowercase_classes", True),
            strict_classes=norm_cfg.get("strict_classes", False),
            enable_alias_map=norm_cfg.get("enable_alias_map", True),
        )
        # Statistics module
        self.stats = DatasetStatistics()

    # =====================================================================
    # SINGLE DATASET BUILD (HARMONIZED)
    # =====================================================================
    def build(self, dataset: Dict[str, Any]) -> DatasetCanonical:
        """
        Build a fully canonical dataset.

        Assumes:
            dataset = CategoryHarmonizer.unify(parsed_list)

        Steps:
            1. Validate structure
            2. Validate / correct bounding boxes
            3. Normalize categories & paths
            4. Compute statistics
            5. Perform splitting
            6. Return DatasetCanonical
        """

        # ------------------------------------------------------------
        # 1 — Structural validation
        # ------------------------------------------------------------
        self.structure_validator.validate(dataset)

        # ------------------------------------------------------------
        # 2 — BBox validation
        # ------------------------------------------------------------
        bbox_errors = self.bbox_validator.validate(dataset)
        report = ValidationReport(bbox_errors=bbox_errors)

        if not self.strict and bbox_errors:
            self.bbox_corrector.correct(dataset)

        if self.strict and bbox_errors:
            report.raise_if_errors()

        # ------------------------------------------------------------
        # 3 — Normalization (paths + categories)
        # ------------------------------------------------------------
        normalized = self.normalizer.normalize(dataset, root=self.root)

        # ------------------------------------------------------------
        # 4 — Statistics computation
        # ------------------------------------------------------------
        normalized["statistics"] = self.stats.compute(normalized)

        # ------------------------------------------------------------
        # 5 — Train/Val/Test splitting
        # ------------------------------------------------------------
        # splitter = SplitStrategyFactory.create(self.config)
        # normalized["splits"] = splitter.split(
        #     normalized["images"],
        #     normalized["annotations"],
        #     normalized["categories"],
        # )

        normalized['info'] = self.config['info']
        normalized['licenses'] = self.config['licenses']
        # ------------------------------------------------------------
        # 6 — Final canonical packaging
        # ------------------------------------------------------------
        return DatasetCanonical(normalized, root=self.root)
