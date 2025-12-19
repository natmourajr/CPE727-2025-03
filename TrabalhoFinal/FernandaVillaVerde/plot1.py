import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

json_path = Path(
    "/home/ferna/CPE727-2025-03/Seminarios/6 - RNN/final_state_lstm_radam.json"
)

with open(json_path, "r") as f:
    data = json.load(f)

# Épocas
epochs = np.arange(1, len(data) + 1)

# ELBO salvo no JSON (negativo)
elbo = np.array([row["micro_elbo_vt"] for row in data])

# ✅ NELBO = |ELBO|  → sempre positivo
nelbo = np.abs(elbo)

# Early stopping correto
best_epoch = 126
best_nelbo = nelbo[best_epoch - 1]

plt.figure(figsize=(8, 4))
plt.plot(epochs, nelbo, lw=2, label="NELBO (teste)")

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
plt.title("Evolução do NELBO — LSTM RAdam (mudança de estado)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
