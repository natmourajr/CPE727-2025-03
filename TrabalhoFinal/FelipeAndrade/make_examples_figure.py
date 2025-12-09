import random
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

IMAGES_ROOT = BASE_DIR / "data" / "crops" / "images"
OUT_DIR = BASE_DIR / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Classes 
CLASSES = [
    "watertank",
    "pool",
    "potted_plant",
    "tire",
    "plastic_bag",
    "bucket",
]

N_PER_CLASS = 4  # quantas imagens por classe

def main():
    random.seed(123)

    all_paths = []
    all_labels = []

    for cls in CLASSES:
        cls_dir = IMAGES_ROOT / cls
        files = sorted(cls_dir.glob("*.jpg"))

        if len(files) == 0:
            print(f"Aviso: sem imagens encontradas para a classe '{cls}' em {cls_dir}")
            continue

        # escolhe N_PER_CLASS imagens aleatórias (ou todas, se tiver menos)
        chosen = random.sample(files, min(N_PER_CLASS, len(files)))
        all_paths.extend(chosen)
        all_labels.extend([cls] * len(chosen))

    n_rows = len(CLASSES)
    n_cols = N_PER_CLASS

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 2.0 * n_rows),
    )

    # Se for uma classe só, axes vira 1D – vamos padronizar para 2D
    if n_rows == 1:
        axes = [axes]

    idx = 0
    for row_idx, cls in enumerate(CLASSES):
        # pega apenas os índices dessa classe
        cls_indices = [i for i, label in enumerate(all_labels) if label == cls]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if col_idx >= len(cls_indices):
                continue

            img_path = all_paths[cls_indices[col_idx]]
            img = Image.open(img_path).convert("RGB")

            ax.imshow(img)
            if col_idx == 0:
                # Nome da classe só na primeira coluna da linha
                ax.set_title(cls, fontsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / "exemplos_crops_mbgv2.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva em: {out_path}")

if __name__ == "__main__":
    main()
