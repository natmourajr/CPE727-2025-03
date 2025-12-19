import json
import pandas as pd
from pathlib import Path

# ==========================
# CONFIGURAÇÃO
# ==========================

MODELS = {
    "Deep":                 ("final_state_deep_.json", 42, 92),
    "LSTM":                 ("final_state_lstm_.json", 19, 69),
    "GRU":                  ("final_state_gru_.json", 22, 72),
    "BiLSTM":               ("final_state_bilstm_.json", 61, 111),
    "BiGRU":                ("final_state_bigru_.json", 55, 105),
    "BiLSTM_SC":            ("final_state_bilstm_warmup_scheduler.json", 11, 61),
    "BiLSTM_RAdam":         ("final_state_lstm_radam.json", 54, 104),
    "BiLSTM_varDropout":    ("final_state_bilstm_vardropout.json", 3, 53),
    "BiLSTM_Diffusion":     ("final_state_bilstm_diffusion.json", 6, 56),
    "BiLSTM_LayerNorm":     ("final_state_bilstm_layernorm.json", 144, 194),
    "BiLSTM_AdamW":         ("final_state_bilstm_adamw.json", 56, 106),
    "LSTM_Adam":            ("final_state_lstm_radam.json", 126, None),
}

BASE_DIR = Path("/home/ferna/CPE727-2025-03/Seminarios/6 - RNN")

FIELDS = {
    "MSE": "micro_mse_vt",
    "MSE_std": "micro_se_vt",
    "NLL": "micro_nll_vt",
    "ELBO": "micro_elbo_vt",
    "Coverage": "micro_cov_vt",
}

# ==========================
# EXTRAÇÃO
# ==========================

rows = []

for model, (fname, best_epoch, early_stop) in MODELS.items():
    path = BASE_DIR / fname
    if not path.exists():
        print(f"[WARN] Arquivo não encontrado: {fname}")
        continue

    with open(path) as f:
        data = json.load(f)

    idx = best_epoch - 1  # epochs começam em 1

    row = {
        "Model": model,
        "Best_epoch": best_epoch,
        "Early_stopping": early_stop,
    }

    for col, key in FIELDS.items():
        row[col] = data[idx].get(key, None)

    rows.append(row)

df = pd.DataFrame(rows)

# ==========================
# SAÍDA
# ==========================

df.to_csv("best_models_micro_with_std.csv", index=False)

print("\nTabela final (micro, vt):\n")
print(df.to_string(index=False))
