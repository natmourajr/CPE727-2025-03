import matplotlib.pyplot as plt
import numpy as np
import os

# Caminho para salvar a figura (ajuste se necessário)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
OUT_PATH = os.path.join(ASSETS_DIR, "macro_f1_barplot.png")

# Modelos e Macro-F1 no teste
models = [
    "CNN base",
    "CNN reg.",
    "AE + MLP",
]
macro_f1 = [
    0.8742,  # CNN base (supervisionada)
    0.6954,  # CNN com regularização
    0.6802,  # Autoencoder + MLP
]

x = np.arange(len(models))

plt.figure(figsize=(5.5, 3.5))  # tamanho bom para slide beamer

bars = plt.bar(x, macro_f1)

# Deixar as barras com valores explícitos em cima
for i, v in enumerate(macro_f1):
    plt.text(
        x[i],
        v + 0.01,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.xticks(x, models)
plt.ylim(0.0, 1.0)
plt.ylabel("Macro-F1 (teste)")
plt.title("Comparação de Macro-F1 entre modelos")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Figura salva em: {OUT_PATH}")
