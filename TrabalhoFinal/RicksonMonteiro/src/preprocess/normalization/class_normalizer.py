from __future__ import annotations
from typing import Dict, List, Optional


class ClassNormalizer:
    """
    Normalize class names using a canonical mapping (aliases).

    Responsibilities:
        - Map alias names to canonical class names
        - Lowercase normalization (optional)
        - Whitespace trimming
        - Handle unseen classes (optional pass-through)
        - Provide a deterministic category_id mapping

    Example alias mapping:
        {
            "coccolith": ["cocco", "cocc.", "cocolite"],
            "foram": ["foraminifera", "foram.", "foraminífero"]
        }

    Usage:
        normalizer = ClassNormalizer(alias_map)
        canonical_name = normalizer.normalize_name("foram.")
        categories = normalizer.build_categories_list()
    """

    def __init__(
        self,
        alias_map: Optional[Dict[str, List[str]]] = None,
        lowercase: bool = True,
        strict: bool = False,
    ) -> None:
        """
        Args:
            alias_map: dict mapping canonical -> list of aliases
            lowercase: apply lowercasing to all class names
            strict: if True, unknown classes raise an error
        """
        self.alias_map = alias_map or {}
        self.lowercase = lowercase
        self.strict = strict

        # Precompute a fast reverse lookup:
        # alias_lookup["foram."] = "foram"
        self.alias_lookup = self._build_reverse_alias_map()

        # Storage for classes encountered in the dataset
        self.class_set: Dict[str, int] = {}

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _build_reverse_alias_map(self) -> Dict[str, str]:
        """Build alias -> canonical lookup from canonical -> alias list."""
        lookup = {}

        for canonical, aliases in self.alias_map.items():
            c_name = canonical.strip().lower() if self.lowercase else canonical.strip()
            lookup[c_name] = canonical  # canonical name maps to itself

            for alias in aliases:
                a_clean = alias.strip().lower() if self.lowercase else alias.strip()
                lookup[a_clean] = canonical

        return lookup

    # ----------------------------------------------------------------------
    # Public API: normalize a single class name
    # ----------------------------------------------------------------------

    def normalize_name(self, name: str) -> str:
        """Normalize a single class name using alias lookup."""
        if not isinstance(name, str):
            raise ValueError(f"Class name must be a string: {name}")

        clean = name.strip()
        if self.lowercase:
            clean = clean.lower()

        # If present in alias table → return canonical
        if clean in self.alias_lookup:
            return self.alias_lookup[clean]

        # If canonical but not lowercase-processed
        if clean in self.alias_map:
            return clean

        # Unknown class
        if self.strict:
            raise KeyError(f"Unknown class name '{name}' (normalized='{clean}')")

        # Pass-through mode → use as-is (but cleaned)
        return clean

    # ----------------------------------------------------------------------
    # Public API: apply normalization to categories
    # ----------------------------------------------------------------------

    def normalize_categories(self, categories: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize categories list.

        Args:
            categories: list of {id, name}

        Returns:
            New list with canonical names.
        """
        normalized = []

        for cat in categories:
            original_name = cat["name"]
            canonical_name = self.normalize_name(original_name)
            normalized.append({
                "id": cat["id"],  # IDs still raw; dataset_normalizer will reassign
                "name": canonical_name,
            })

        return normalized

    # ----------------------------------------------------------------------
    # Track all classes observed (for building canonical category IDs)
    # ----------------------------------------------------------------------

    def register_class(self, name: str) -> None:
        """Register a class name found during parsing/normalization."""
        canonical = self.normalize_name(name)
        if canonical not in self.class_set:
            self.class_set[canonical] = len(self.class_set)

    def build_categories_list(self) -> List[Dict[str, str]]:
        """
        Build canonical categories list using observed classes.

        Returns:
            [
                {"id": 0, "name": "coccolith"},
                {"id": 1, "name": "foram"},
                ...
            ]
        """
        return [
            {"id": idx, "name": name}
            for name, idx in sorted(self.class_set.items(), key=lambda x: x[1])
        ]
