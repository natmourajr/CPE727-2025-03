import json
import matplotlib.pyplot as plt
from pathlib import Path

# --- Caminho do JSON do modelo campeão ---
json_path = Path(
    "/home/ferna/CPE727-2025-03/Seminarios/6 - RNN/final_rebuild_bilstm_diffusion.json"
)

# --- Carregar dados ---
with open(json_path, "r") as f:
    data = json.load(f)

# --- Extrair ELBO por época ---
epochs = []
elbo = []

for i, row in enumerate(data):
    # i começa em 0 → época começa em 1
    epochs.append(i + 1)
    elbo.append(row.get("micro_elbo_v"))

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(epochs, elbo, lw=2)
plt.axvline(
    x=49, color="red", linestyle="--", linewidth=1,
    label="Melhor época (49)"
)

plt.xlabel("Época")
plt.ylabel("ELBO (validação)")
plt.title("Evolução do NELBO — BiLSTM Difusão")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
