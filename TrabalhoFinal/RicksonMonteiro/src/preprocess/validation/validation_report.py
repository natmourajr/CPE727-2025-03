from __future__ import annotations
from typing import List, Dict, Any


class ValidationReport:
    """
    Collects, summarizes and raises validation issues detected during
    dataset validation steps.

    This object is returned by validators and consumed by DatasetBuilder.

    Responsibilities:
        - Hold all validation warnings/errors
        - Provide summary statistics
        - Produce human-readable report lines
        - Raise exceptions in strict mode
    """

    def __init__(
        self,
        bbox_errors: List[Dict[str, Any]] | None = None,
        structure_errors: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.bbox_errors = bbox_errors or []
        self.structure_errors = structure_errors or []

    # ----------------------------------------------------------------------
    # Error aggregation
    # ----------------------------------------------------------------------

    @property
    def total_errors(self) -> int:
        return len(self.bbox_errors) + len(self.structure_errors)

    @property
    def has_errors(self) -> bool:
        return self.total_errors > 0

    # ----------------------------------------------------------------------
    # Summary generation
    # ----------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary useful for logging or MLflow."""

        return {
            "total_errors": self.total_errors,
            "bbox_errors": len(self.bbox_errors),
            "structure_errors": len(self.structure_errors),
        }

    def to_lines(self) -> List[str]:
        """
        Produce human-readable lines summarizing all issues.
        Useful for console logging or writing to a validation.txt file.
        """
        lines = []

        # --- BBox errors ----------------------------------------------------
        if self.bbox_errors:
            lines.append(f"BBox Errors: {len(self.bbox_errors)}")
            for err in self.bbox_errors[:20]:
                lines.append(
                    f"  - {err['reason']} | bbox={err['bbox']} | \
                        ann_id={err['annotation'].get('id')}"
                )
            if len(self.bbox_errors) > 20:
                lines.append(f"  ... {len(self.bbox_errors)-20} more")

        # --- Structure errors ----------------------------------------------
        if self.structure_errors:
            lines.append(f"Structure Errors: {len(self.structure_errors)}")
            for err in self.structure_errors[:20]:
                lines.append(f"  - {err}")

        return lines

    # ----------------------------------------------------------------------
    # Exception handling (strict mode)
    # ----------------------------------------------------------------------

    def raise_if_errors(self) -> None:
        """
        Raise a ValueError if any validation errors exist.
        Used when DatasetBuilder(strict=True).
        """
        if not self.has_errors:
            return

        msg = "\n".join([
            "Dataset validation failed:",
            *self.to_lines()
        ])
        raise ValueError(msg)


    def __str__(self) -> str:
        return "\n".join(self.to_lines()) or "No validation issues found."
