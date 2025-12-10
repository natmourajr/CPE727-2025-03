import zipfile
import shutil
from pathlib import Path
from typing import Iterator
import json
import logging
import io
import unicodedata
import re
from typing import Dict, List, Any, Iterator
from dataclasses import dataclass


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass(frozen=True)
class RawExperiment:
    """
    Represents a single raw experiment extracted from any data source.

    experiment_id_raw   → nome original (como veio do ZIP ou da fonte)
    experiment_id_clean → nome sanitizado (usado em paths internos)
    
    This object:
        - Exposes ONLY file paths
        - Performs NO parsing or validation
        - Is immutable (frozen=True)
    """

    experiment_id_raw: str               
    experiment_id_clean: str             
    root: Path                           
    image_files: List[Path]              
    label_files: List[Path]              
    class_dict: Dict[str, int]           
    metadata: Dict[str, Any]             


def clean_experiment_id(name: str) -> str:
    """Normalize experiment ID for safe filesystem usage (Windows/Linux)."""

    # Remove acentos
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()

    # Trocar qualquer caractere não alfanumérico por "_"
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)

    # Remover múltiplos underscores
    name = re.sub(r"_+", "_", name).replace("-", "_").strip()

    return name


class ZipLoader():
    """
    ZIP Loader with full support for:
    - flat ZIPs
    - multi-directory ZIPs
    - nested ZIPs

    NOW WITH EXPERIMENT NAME NORMALIZATION:
    - experiment_id_raw  → nome original do diretório/zip
    - experiment_id_clean → nome seguro para filesystem (usado nos paths)
    """

    def __init__(self, zip_path: Path, extract_to: Path, overwrite: bool = False, multi_experiment: bool = False):
        self.zip_path = Path(zip_path)
        self.extract_to = Path(extract_to)
        self.overwrite = overwrite
        self.multi_experiment = multi_experiment

        if not self.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        if not zipfile.is_zipfile(self.zip_path):
            raise ValueError(f"Not a valid ZIP archive: {zip_path}")

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------

    def _prepare_exp_dir(self, exp_id_raw: str) -> Path:
        """
        Creates persistent extraction folder for each experiment.

        exp_id_raw   → nome original no ZIP
        exp_id_clean → nome sanitizado seguro (filesystem-safe)
        """
        exp_id_clean = clean_experiment_id(exp_id_raw)

        exp_dir = self.extract_to / exp_id_clean

        if self.overwrite and exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)

        exp_dir.mkdir(parents=True, exist_ok=True)

        # Log changes
        if exp_id_clean != exp_id_raw:
            logger.info(f"[ZipLoader] Normalized experiment ID: '{exp_id_raw}' → '{exp_id_clean}'")

        return exp_dir

    def _collect_experiment_from_dir(self, dirpath: Path, exp_id_raw: str) -> RawExperiment:
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
        label_exts = {".txt", ".json", ".xml", ".csv"}

        images = []
        labels = []
        class_dict = {}

        exp_id_clean = clean_experiment_id(exp_id_raw)

        metadata = {
            "source": str(self.zip_path),
            "experiment_id_raw": exp_id_raw,
            "experiment_id_clean": exp_id_clean,
        }

        for p in dirpath.rglob("*"):
            if not p.is_file():
                continue

            ext = p.suffix.lower()

            if ext in image_exts:
                images.append(p)
            elif ext in label_exts:
                if p.name.lower() == "notes.json":
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                class_dict = data
                            elif isinstance(data, list):
                                tmp = {}
                                for item in data:
                                    name = item.get("name") or item.get("label") or item.get("title")
                                    if name:
                                        tmp[name] = item.get("id", len(tmp))
                                class_dict = tmp
                    except Exception as exc:
                        logger.warning(f"Failed to load notes.json at {p}: {exc}")
                else:
                    labels.append(p)

        images.sort()
        labels.sort()

        return RawExperiment(
            experiment_id_raw=exp_id_raw,
            experiment_id_clean=exp_id_clean,
            root=dirpath,
            image_files=images,
            label_files=labels,
            class_dict=class_dict,
            metadata=metadata,
        )

    # -----------------------------------------------------------
    # Main load
    # -----------------------------------------------------------

    def load(self) -> Iterator[RawExperiment]:
        with zipfile.ZipFile(self.zip_path, "r") as zfile:

            # 1. Nested ZIPs --------------------------------------------------
            nested = [n for n in zfile.namelist() if n.lower().endswith(".zip")]
            if nested:
                for member in nested:
                    exp_id_raw = Path(member).stem
                    exp_dir = self._prepare_exp_dir(exp_id_raw)

                    inner_bytes = zfile.read(member)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                        inner_zip.extractall(exp_dir)

                    yield self._collect_experiment_from_dir(exp_dir, exp_id_raw)
                return

            # 2. Multi-directory ZIPs ----------------------------------------
            top_dirs = sorted({
                p.split("/")[0] + "/"
                for p in zfile.namelist()
                if "/" in p
            })

            if top_dirs:
                for dirname in top_dirs:
                    exp_id_raw = Path(dirname).stem
                    exp_dir = self._prepare_exp_dir(exp_id_raw)

                    for info in zfile.infolist():
                        if info.filename.startswith(dirname):
                            rel = Path(info.filename).relative_to(dirname)
                            target = exp_dir / rel
                            target.parent.mkdir(parents=True, exist_ok=True)

                            with zfile.open(info) as src, open(target, "wb") as dst:
                                shutil.copyfileobj(src, dst)

                    yield self._collect_experiment_from_dir(exp_dir, exp_id_raw)
                return

            # 3. Flat ZIP → Single Experiment -------------------------------
            exp_id_raw = self.zip_path.stem
            exp_dir = self._prepare_exp_dir(exp_id_raw)

            zfile.extractall(exp_dir)

            yield self._collect_experiment_from_dir(exp_dir, exp_id_raw)
