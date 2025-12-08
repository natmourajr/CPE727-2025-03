import json
from pathlib import Path
import matplotlib.pyplot as plt

# ---- 1) Caminho do JSON deste modelo ----
BASE_DIR = Path("/home/ferna/CPE727-2025-03/Seminarios/6 - RNN/")
json_path = BASE_DIR / "final_state_bilstm_radam.json"

# ---- 2) Carregar todas as épocas ----
with open(json_path, "r") as f:
    data = json.load(f)   # lista de dicts: 1º epoch = data[0], etc.

# ---- 3) Extrair ELBO por época ----
elbos = [row["micro_elbo_vt"] for row in data]
epochs = list(range(1, len(elbos) + 1))

# Melhor época (pelo log)
best_epoch = 54
best_elbo = elbos[best_epoch - 1]

# ---- 4) Plotar evolução do ELBO ----
plt.figure(figsize=(8, 4))
plt.plot(epochs, elbos, label="NELBO (micro_elbo_vt)")
plt.axvline(best_epoch, linestyle="--", label=f"Melhor época = {best_epoch}")
plt.scatter([best_epoch], [best_elbo])  # marca o ponto ótimo

plt.xlabel("Época")
plt.ylabel("NELBO (VAE tmax)")
plt.title("Evolução do NELBO por época – BiLSTM RAdam")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
