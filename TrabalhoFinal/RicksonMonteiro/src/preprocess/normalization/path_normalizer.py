from pathlib import Path
from typing import Dict, Any


class PathNormalizer:
    """
    Normalizes file_name fields in canonical dataset:
        - Removes Windows backslashes
        - Converts to POSIX slashes
        - Removes any directory prefixes (images/, foo/bar/, etc.)
        - Returns ONLY the filename (e.g. 'D1_1_250_1.jpg')
    """

    def __init__(self, root: Path | None = None) -> None:
        # root não é mais necessário, mas mantemos por compatibilidade
        self.root = Path(root).resolve() if root else None

    # --------------------------------------------------------------
    @staticmethod
    def normalize_filename(name: str) -> str:
        """
        Ensures file_name becomes only '<filename>.<ext>'
        """
        # Convert backslashes → forward slashes
        name = str(name).replace("\\", "/")

        # Remove all directory prefixes
        name = name.split("/")[-1]

        return name

    # --------------------------------------------------------------
    def normalize_dataset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply filename normalization over canonical dataset.
        """

        normalized = dict(data)
        out_images = []

        for img in data.get("images", []):
            clean_name = self.normalize_filename(img["file_name"])

            out_images.append({
                **img,
                "file_name": clean_name,
            })

        normalized["images"] = out_images
        return normalized
