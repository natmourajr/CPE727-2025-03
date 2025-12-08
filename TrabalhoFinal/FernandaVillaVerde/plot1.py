import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

json_path = Path(
    "/home/ferna/CPE727-2025-03/Seminarios/6 - RNN/final_state_bilstm_radam.json"
)

with open(json_path, "r") as f:
    data = json.load(f)

# Épocas (1, 2, ..., N)
epochs = np.arange(1, len(data) + 1)

# ELBO por época (validação) -> campo correto é micro_elbo_vt
elbo = np.array([row["micro_elbo_vt"] for row in data])

# NELBO = - ELBO
nelbo = -elbo

# Melhor época (fixa, vinda do seu early stopping)
best_epoch = 54
best_nelbo = nelbo[best_epoch - 1]

plt.figure(figsize=(8, 4))
plt.plot(epochs, nelbo, lw=2, label="NELBO (validação)")

plt.axvline(
    x=best_epoch,
    color="red",
    linestyle="--",
    linewidth=1,
    label=f"Melhor época (early stopping = {best_epoch})"
)

plt.scatter(
    best_epoch, best_nelbo,
    color="red",
    zorder=3
)

plt.xlabel("Época")
plt.ylabel("NELBO (teste)")
plt.title("Evolução do NELBO — BiLSTM RAdam")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
