import os
import numpy as np
import matplotlib.pyplot as plt

# Caminho base: ajuste se seu .tex estiver em outro lugar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

class_names = [
    "bottle",
    "bucket",
    "dumpster",
    "large_trash_bin",
    "plastic_bag",
    "pool",
    "potted_plant",
    "puddle",
    "small_trash_bin",
    "storm_drain",
    "tire",
    "watertank",
]

# Matrizes de confusão copiadas dos logs (linhas = verdade, colunas = predição)
cm_cnn_base = np.array([
    [  55,    7,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2],
    [   6,  371,    0,    1,    2,    1,   18,    0,    3,    1,   13,   11],
    [   0,    0,    5,    0,    1,    1,    0,    0,    0,    0,    0,    0],
    [   0,    8,    0,   23,    0,    0,    1,    0,    0,    0,    2,    2],
    [   0,    2,    0,    0,   90,    0,    0,    0,    0,    1,    1,    3],
    [   0,    6,    0,    0,    2,  858,    4,    0,    0,    0,    0,   72],
    [   0,   48,    0,    0,    4,    1,  999,    0,    1,   25,   36,   17],
    [   0,    0,    0,    0,    0,    2,    0,   34,    0,    0,    0,    5],
    [   0,    2,    0,    1,    2,    0,    0,    0,   30,    0,    1,    1],
    [   0,    8,    0,    0,    5,    0,   18,    0,    0, 1029,   10,   11],
    [   1,   15,    0,    1,    3,    0,   12,    0,    0,    3,  452,   11],
    [   3,   30,    0,    2,    5,    8,   11,    0,    0,    2,   10, 4204],
])

cm_cnn_reg = np.array([
    [  52,    6,    0,    0,    1,    0,    3,    0,    0,    0,    0,    2],
    [  11,  300,    0,    0,    2,    1,   29,    0,    3,    3,   38,   40],
    [   0,    0,    3,    0,    0,    1,    0,    0,    0,    0,    0,    3],
    [   0,   11,    0,    5,    1,    0,    3,    0,    4,    0,    2,   10],
    [   1,    3,    0,    0,   40,    0,    7,    0,    0,    3,   19,   24],
    [   0,    2,    0,    0,    0,  639,    9,    0,    0,    0,    1,  291],
    [   1,   40,    0,    0,    0,    1, 1015,    0,    0,   11,   46,   17],
    [   0,    0,    0,    0,    0,    2,    2,   23,    0,    0,    0,   14],
    [   0,    5,    0,    4,    2,    0,    0,    0,   13,    0,    2,   11],
    [   0,    4,    0,    0,    0,    0,   57,    0,    0, 1000,   14,    6],
    [   2,   23,    0,    0,    6,    0,   39,    0,    0,    3,  415,   10],
    [   6,   35,    0,    1,    4,    3,   13,    0,    0,    2,   13, 4198],
])

cm_ae_mlp = np.array([
    [  37,   11,    0,    0,    1,    1,    0,    0,    0,    2,    0,   12],
    [   1,  278,    0,    0,    3,   11,   41,    1,    6,    5,   20,   61],
    [   0,    1,    1,    0,    0,    3,    0,    0,    0,    0,    0,    2],
    [   0,    5,    0,    9,    0,    0,    6,    1,    1,    1,    2,   11],
    [   0,    7,    0,    0,   42,    8,    4,    1,    0,    8,   10,   17],
    [   1,    6,    0,    0,    1,  843,   10,    0,    0,    5,    2,   74],
    [   0,   29,    0,    0,    0,   10, 1003,    0,    0,   29,   35,   25],
    [   0,    0,    0,    0,    0,    0,    7,   23,    0,    3,    0,    8],
    [   0,    6,    0,    0,    0,    3,    2,    0,   19,    1,    2,    4],
    [   0,    5,    0,    0,    0,    1,   59,    2,    0,  981,   11,   22],
    [   2,   23,    0,    0,    0,    3,   50,    2,    0,   16,  349,   53],
    [   1,   30,    0,    3,    3,   46,   26,    0,    1,   17,   22, 4126],
])


def plot_confusion(cm, class_names, title, out_path, normalize=True):
    """
    cm: matriz de confusão (numpy array)
    normalize: se True, normaliza por linha (recall por classe)
    """
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right", fontsize=7)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=7)

    plt.xlabel("Predito", fontsize=10)
    plt.ylabel("Verdadeiro", fontsize=10)
    plt.title(title, fontsize=11)

    # grade leve
    plt.grid(which="both", linestyle=":", linewidth=0.3, color="gray", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figura salva em: {out_path}")


# Gerar as três figuras
plot_confusion(
    cm_cnn_base,
    class_names,
    "Matriz de confusão – CNN base",
    os.path.join(ASSETS_DIR, "confusion_cnn_base.png"),
)

plot_confusion(
    cm_cnn_reg,
    class_names,
    "Matriz de confusão – CNN regularizada",
    os.path.join(ASSETS_DIR, "confusion_cnn_reg.png"),
)

plot_confusion(
    cm_ae_mlp,
    class_names,
    "Matriz de confusão – AE + MLP",
    os.path.join(ASSETS_DIR, "confusion_ae_mlp.png"),
)
