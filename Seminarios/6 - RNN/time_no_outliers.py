import json
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# MODELOS QUE PRECISAM FILTRO
# ----------------------------
FILTER_MODELS = {
    "final_state_bilstm_adamw",
    "final_rebuild_bilstm_radam",
    "final_state_bilstm_radam",
}

# ----------------------------
# Utilidades
# ----------------------------
def extract_epoch_times(data):
    return np.array(
        [e["epoch_time"] for e in data if isinstance(e, dict) and "epoch_time" in e],
        dtype=float
    )

def filter_outliers_mad(x, z=3.5):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return x
    rz = 0.6745 * (x - med) / mad
    return x[np.abs(rz) <= z]

# ----------------------------
# Pipeline principal
# ----------------------------
rows = []

runs_dir = Path("/home/ferna/CPE727-2025-03/Seminarios/6 - RNN")

for json_file in sorted(runs_dir.glob("*.json")):
    model = json_file.stem

    with open(json_file, "r") as f:
        data = json.load(f)

    epoch_times = extract_epoch_times(data)
    n_epochs = len(epoch_times)

    # --- decisão cirúrgica ---
    if model in FILTER_MODELS:
        epoch_used = filter_outliers_mad(epoch_times)
        method = "MAD (outliers removidos)"
    else:
        epoch_used = epoch_times
        method = "sem filtragem"

    mean_s = epoch_used.mean()
    std_s = epoch_used.std(ddof=1) if len(epoch_used) > 1 else 0.0
    median_s = np.median(epoch_used)

    # tempo total:
    if model in FILTER_MODELS:
        # robusto: mediana × N original
        total_min = median_s * n_epochs / 60
    else:
        # normal: soma direta
        total_min = epoch_times.sum() / 60

    rows.append({
        "Modelo": model,
        "Épocas": n_epochs,
        "Tempo total (min)": round(total_min, 3),
        "Tempo por época (s)": f"{mean_s:.2f} ± {std_s:.2f}",
        "Mediana por época (s)": round(median_s, 2),
        "Tratamento": method
    })

df = pd.DataFrame(rows).sort_values("Tempo total (min)")
df.to_csv("training_time_summary_selective.csv", index=False)

print(df)
print("\nCSV salvo em: training_time_summary_selective.csv")


import pandas as pd

df = pd.read_csv("training_time_summary_selective.csv")  # ou o nome do seu CSV atual

col = "Tempo total (min)"
df[col] = pd.to_numeric(df[col], errors="coerce")

# Corrige valores tipo 618, 1255, 3908... -> 0.618, 1.255, 3.908...
df.loc[df[col] > 100, col] = df.loc[df[col] > 100, col] / 1000.0

# (opcional) arredondar
df[col] = df[col].round(3)

df.to_csv("training_time_summary_selective_FIXED.csv", index=False)
print(df[["Modelo", "Épocas", "Tempo total (min)"]])
