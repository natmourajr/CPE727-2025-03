from __future__ import annotations
from typing import Dict, List, Optional, Any


class ClassGrouper:
    """
    Assign classes into taxonomic/morphological groups.

    Supports two YAML formats:

    1) Group → list of classes
        fossil:
          - alga
          - bentonico

    2) Class → {group: ...}
        alga:
          group: fossil

    All formats are unified into:
        class_to_group: {class → group}
        group_to_classes: {group → [classes]}
    """

    def __init__(
        self,
        group_mapping: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> None:
        self.strict = strict
        self.group_mapping_raw = group_mapping or {}

        # Build unified mapping
        self.class_to_group = self._build_class_to_group_map(self.group_mapping_raw)
        self.group_to_classes = self._build_group_to_classes_map(self.class_to_group)

    # ======================================================================
    # INTERNAL — NORMALIZATION OF YAML FORMAT
    # ======================================================================

    def _build_class_to_group_map(self, raw: Dict[str, Any]) -> Dict[str, str]:
        """
        Builds: class_name → group_name

        Accepts two YAML formats:
            group: [classes...]
            class: {group: "name"}
        """
        normalized: Dict[str, str] = {}

        for key, value in raw.items():

            # --------------------------------------------------------------
            # CASE A — group: [class1, class2...]
            # --------------------------------------------------------------
            if isinstance(value, list):
                group_name = key
                for cls in value:
                    normalized[cls] = group_name

            # --------------------------------------------------------------
            # CASE B — class: {group: "name"}
            # --------------------------------------------------------------
            elif isinstance(value, dict) and "group" in value:
                group_name = value["group"]
                normalized[key] = group_name

            else:
                raise ValueError(
                    f"[ClassGrouper] Invalid entry in group_map: {key}: {value}. "
                    f"Expected list (children) or dict with 'group' field."
                )

        return normalized

    def _build_group_to_classes_map(self, class_to_group: Dict[str, str]) -> Dict[str, List[str]]:
        """Reverse mapping: group → [classes]."""
        reverse: Dict[str, List[str]] = {}

        for cls, grp in class_to_group.items():
            reverse.setdefault(grp, []).append(cls)

        # Deterministic ordering helps reproducibility
        for grp in reverse:
            reverse[grp] = sorted(reverse[grp])

        return reverse

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def get_group(self, class_name: str) -> Optional[str]:
        """
        Return the group for a canonical class name.
        """
        if class_name in self.class_to_group:
            return self.class_to_group[class_name]

        if self.strict:
            raise KeyError(f"[ClassGrouper] Class '{class_name}' has no defined group.")
        return None

    def get_group_id(self, class_name: str) -> Optional[int]:
        """Deterministic numeric id based on alphabetical group ordering."""
        grp = self.get_group(class_name)
        if grp is None:
            return None
        all_groups = sorted(self.group_to_classes.keys())
        return all_groups.index(grp)

    def get_all_groups(self) -> List[str]:
        return sorted(self.group_to_classes.keys())

    def get_classes_in_group(self, group_name: str) -> List[str]:
        return self.group_to_classes.get(group_name, [])
