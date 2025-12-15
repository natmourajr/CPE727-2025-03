import json
import numpy as np
import pandas as pd
from pathlib import Path

# pasta onde estão os JSONs
RUNS_DIR = Path("/home/ferna/CPE727-2025-03/Seminarios/6 - RNN")   # ajuste se necessário

rows = []

for json_file in RUNS_DIR.glob("*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

    # assume lista de épocas (como no seu caso)
    epoch_times = [e["epoch_time"] for e in data if "epoch_time" in e]

    if len(epoch_times) == 0:
        print(f"[WARN] {json_file.name} sem epoch_time")
        continue

    epoch_times = np.array(epoch_times)

    total_s = epoch_times.sum()
    mean_s = epoch_times.mean()
    std_s = epoch_times.std(ddof=1)
    n_epochs = len(epoch_times)

    rows.append({
        "Modelo": json_file.stem,
        "Épocas": n_epochs,
        "Tempo total (min)": total_s / 60,
        "Tempo por época (s)": f"{mean_s:.2f} ± {std_s:.2f}"
    })

df = pd.DataFrame(rows)

# ordena pelo tempo total
df = df.sort_values("Tempo total (min)")

# salva CSV
df.to_csv("training_time_summary.csv", index=False)

print("Resumo salvo em training_time_summary.csv")
print(df)

