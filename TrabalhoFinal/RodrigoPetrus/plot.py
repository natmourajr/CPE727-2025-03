import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

json_path = Path(
    "/home/ferna/CPE727-2025-03/Seminarios/6 - RNN/final_rebuild_bilstm_diffusion.json"
)

with open(json_path, "r") as f:
    data = json.load(f)

epochs = np.arange(1, len(data) + 1)
nelbo = -np.array([row["micro_elbo_v"] for row in data])

best_epoch = 133  # EARLY STOPPING
best_nelbo = nelbo[best_epoch - 1]

plt.figure(figsize=(8, 4))
plt.plot(epochs, nelbo, lw=2)

plt.axvline(
    x=best_epoch,
    color="red",
    linestyle="--",
    linewidth=1,
    label=f"Melhor época (early stopping = {best_epoch})"
)

plt.scatter(
    best_epoch, best_nelbo,
    color="red", zorder=3
)

plt.xlabel("Época")
plt.ylabel("NELBO (teste)")
plt.title("Evolução do NELBO — BiLSTM Difusão")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
