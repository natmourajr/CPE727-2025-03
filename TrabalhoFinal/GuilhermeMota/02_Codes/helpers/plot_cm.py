import matplotlib.pyplot as plt
import numpy as np
import os



def plot_confusion_matrix(cm, labels, save_path=None):

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predição")
    ax.set_ylabel("Verdadeiro")
    plt.title("Matriz Confusão")

    cm_sum = cm.sum()
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            pct = value / cm_sum * 100
            text = f"{value}\n({pct:.1f}%)"

            r, g, b, _ = im.cmap(im.norm(value))
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = "black" if luminance > 0.5 else "white"

            ax.text(j, i, text, ha="center", va="center",
                    color=text_color, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()