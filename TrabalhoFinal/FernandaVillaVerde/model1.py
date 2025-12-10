import json
import pandas as pd
from pathlib import Path

# ---- 1) Função para extrair métricas da melhor época ----
def extract_metrics(json_file: str | Path, best_epoch: int):
    json_file = Path(json_file)
    with open(json_file, "r") as f:
        data = json.load(f)

    idx = best_epoch - 1
    if idx < 0 or idx >= len(data):
        raise IndexError(f"best_epoch={best_epoch} fora do range para {json_file.name} (len={len(data)})")

    row = data[idx]

    return {
        "ELBO":    row.get("micro_elbo_vt", None),
        "MSE":     row.get("micro_mse_s", None),
        "NLL":     row.get("micro_nll_vt", None),
        "Cov90":   row.get("micro_cov_vt", None),
        "Width90": row.get("micro_width_vt", None),
    }


# ---- 2) Mapear modelos -> (arquivo json, melhor época) ----
# Ajuste BASE_DIR se necessário
BASE_DIR = Path("/home/ferna/CPE727-2025-03/Seminarios/6 - RNN")  

models_info = {
    "DEEP": ("final_state_deep_.json", 42),
    "LSTM": ("final_state_lstm_.json", 19),
    "GRU": ("final_state_gru_.json", 22),
    "BiLSTM": ("final_state_bilstm_.json", 61),
    "BiGRU": ("final_state_bigru_.json", 55),
    "BiLSTM (GRU Fuser) Warmup/Sched.": ("final_state_bilstm_warmup_scheduler.json", 11),
    "BiLSTM (GRU Fuser) RAdam": ("final_state_bilstm_radam.json", 54),
    "BiLSTM (GRU Fuser) VarDrop 0.2": ("final_state_bilstm_vardropout.json", 3),
    "BiLSTM (GRU Fuser) Difusão": ("final_state_bilstm_diffusion.json", 6),
    "BiLSTM (GRU Fuser) LayerNorm": ("final_state_bilstm_layernorm.json", 144),
    "BiLSTM (GRU Fuser) AdamW": ("final_state_bilstm_adamw.json", 56),
}


# ---- 3) (Novo) Desvio padrão APENAS do MSE micro (tirado dos logs) ----
# Preencha esses valores com os "±" do log para cada modelo.
micro_mse_std = {
    "DEEP": 0.011309,
    "LSTM": 0.000252,
    "GRU": 0.000753,
    "BiLSTM": 0.000397,
    "BiGRU": 0.000425,
    "BiLSTM (GRU Fuser) Warmup/Sched.": 0.000189,
    "BiLSTM (GRU Fuser) RAdam": 0.000182,
    "BiLSTM (GRU Fuser) VarDrop 0.2": 0.001096,
    "BiLSTM (GRU Fuser) Difusão": 0.000593,
    "BiLSTM (GRU Fuser) LayerNorm": 0.000630,
    "BiLSTM (GRU Fuser) AdamW": 0.000528,
}


# ---- 4) Montar DataFrame com resultados ----
rows = {}
for model_name, (fname, best_epoch) in models_info.items():
    json_path = BASE_DIR / fname
    metrics = extract_metrics(json_path, best_epoch)
    rows[model_name] = metrics

df = pd.DataFrame.from_dict(rows, orient="index")
df.index.name = "Modelo"

# adiciona coluna de desvio do MSE micro
df["MSE_micro_std"] = df.index.map(micro_mse_std.get)

print("✅ Tabela bruta (sem formatação):")
print(df)

# ---- 5) Função segura para formatar média ± desvio ----
def fmt_mean_std(mean, std):
    if mean is None or pd.isna(mean):
        return "--"
    if std is None or pd.isna(std):
        # só a média, sem desvio
        return f"{mean:.3f}"
    # média ± desvio
    return f"{mean:.3f} ± {std:.3f}"


# ---- 6) Construir DataFrame formatado para LaTeX ----
df_fmt = pd.DataFrame(index=df.index)
df_fmt["ELBO"] = df["ELBO"].round(3)
df_fmt["MSE (micro)"] = [
    fmt_mean_std(m, s) for m, s in zip(df["MSE"], df["MSE_micro_std"])
]
df_fmt["NLL"] = df["NLL"].round(3)
df_fmt["Cov90"] = df["Cov90"].round(3)
df_fmt["Width90"] = df["Width90"].round(3)

# ordenar pelo ELBO (decrescente)
df_fmt = df_fmt.sort_values("ELBO", ascending=False)

print("\n✅ Tabela formatada (ordenada por ELBO):")
print(df_fmt)

# ---- 7) Gerar LaTeX pronto para Beamer ----
latex_table = df_fmt.to_latex(
    float_format="%.3f",
    column_format="lccccc",
    escape=True,
)

out_path = BASE_DIR / "tabela_resultados_mse_micro_std.tex"
with open(out_path, "w") as f:
    f.write(latex_table)

print(f"\n📄 Tabela LaTeX salva em: {out_path}")
