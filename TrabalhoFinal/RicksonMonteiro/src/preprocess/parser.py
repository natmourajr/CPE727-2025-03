from __future__ import annotations
from pathlib import Path
from PIL import Image

from typing import Any, Dict, List, Optional, Iterable


class Parser():
    """
    Robust YOLO annotation parser.
    
    Adds experiment_id metadata to each image.

    Expected directory structure:
        root/
            images/
            labels/
            classes.txt

    Output keys in "images":
        - id
        - file_name (relative)
        - width
        - height
        - experiment_id   <-- NEW
    """
    def __init__(self, root: Path, experiment_id: str,
                 classes: Optional[Iterable[str]] = None):
       
        self.root = Path(root)
        self._classes = list(classes) if classes is not None else None
        self.experiment_id = experiment_id

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def parse(self) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # Step 1 — Load categories
        # ------------------------------------------------------------------
        categories = self._load_categories()

        # ------------------------------------------------------------------
        # Step 2 — Index images
        # ------------------------------------------------------------------
        image_index = self._index_images()

        # ------------------------------------------------------------------
        # Step 3 — Parse annotations
        # ------------------------------------------------------------------
        labels_dir = self.root / "labels"
        txt_files = sorted(labels_dir.glob("*.txt"))

        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        ann_id = 1
        img_id = 1

        # experiment_id is captured from BaseParser: self.experiment_id
        exp_id = self.experiment_id

        for txt_file in txt_files:
            stem = txt_file.stem
            image_path = image_index.get(stem)

            if image_path is None:
                self._log_warning(
                    f"No corresponding image for label: {txt_file}"
                )
                continue
            # Load image dimensions
            # try:
            with Image.open(image_path) as im:
                width, height = im.size
            # except Exception:

            #     width, height = None, None
            #     self._log_warning(f"Failed to read image size: {image_path}")

            # Register image
            images.append({
                "id": img_id,
                "file_name": str(image_path.relative_to(self.root)),
                "width": width,
                "height": height,
                "experiment_id": exp_id
            })

            # Parse annotations
            parsed_anns = self._parse_label_file(
                txt_file=txt_file,
                image_id=img_id,
                ann_start_id=ann_id,
                img_width=width,
                img_height=height
            )
            annotations.extend(parsed_anns)

            ann_id += len(parsed_anns)
            img_id += 1

        # ------------------------------------------------------------------
        # Step 4 — COCO-like output
        # ------------------------------------------------------------------
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    def _index_images(self) -> Dict[str, Path]:
        images_dir = self.root / "images"
        if not images_dir.exists():
            raise RuntimeError(f"Images directory not found: {images_dir}")

        index = {}
        for img in images_dir.rglob("*"):
            if img.suffix.lower() in self.IMAGE_EXTENSIONS:
                index[img.stem] = img
        return index

    def _load_categories(self) -> List[Dict[str, Any]]:
        categories: List[Dict[str, Any]] = []

        # If externally provided
        if self._classes:
            for i, name in enumerate(self._classes):
                categories.append({"id": i, "name": name})
            return categories

        # Otherwise load from classes.txt
        classes_file = self.root / "classes.txt"
        if classes_file.exists():
            for i, line in enumerate(classes_file.read_text(encoding="utf-8").splitlines()):
                name = line.strip()
                if name:
                    categories.append({"id": i, "name": name}) # +1 to convert to COCO format IDs (0 -> 1)
        else:
            self._log_warning("No classes provided and classes.txt not found.")

        return categories

    def _parse_label_file(self, txt_file: Path, image_id: int, ann_start_id: int, img_width:int, img_height:int) -> List[Dict[str, Any]]:
        annotations = []

        lines = [
            line.strip()
            for line in txt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        ann_id = ann_start_id

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                self._log_warning(f"Malformed line in {txt_file}: '{line}'")
                continue

            cls_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # Convert normalized → absolute
            x_center_abs = x_center_norm * img_width
            y_center_abs = y_center_norm * img_height
            
            w_abs = w_norm * img_width
            h_abs = h_norm * img_height

            xmin = max(0, x_center_abs - w_abs / 2)
            ymin = max(0, y_center_abs - h_abs / 2)

            w_abs = min(w_abs, img_width - xmin)
            h_abs = min(h_abs, img_height - ymin)

            area = w_abs * h_abs
            bbox = [xmin, ymin, w_abs, h_abs] 

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })

            ann_id += 1

        return annotations


    def _log_warning(self, msg: str):
        print(f"[Parser:WARNING] {msg}")
