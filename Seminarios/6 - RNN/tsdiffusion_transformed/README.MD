# TS Diffusion (Jump-ODE + Difusão)
Implementação em PyTorch de um modelo de difusão para séries temporais com encoder Jump-ODE, self-attention e embeddings temporais híbridos. Inclui pipelines de treino, imputação e previsão recursiva usando DataFrames (ex.: exportados da Cognite) com colunas de séries, atributos estáticos e janelas de predição de estado.

## Dependências
- Python 3.10+
- PyTorch (GPU opcional)
- torchdiffeq
- pandas, numpy, scikit-learn, scipy
- cognite-sdk (para ingestão direta)
- matplotlib/seaborn (apenas para gráficos de métricas)

Instale as bibliotecas com `pip install -r requirements.txt` na raiz do projeto ou instale manualmente os pacotes listados acima.


## Estrutura do pacote
|`ts_diffusion_transformed.py` | modelo TSDiffusion (encoder linear + JumpODEEncoder + cabeças de perda, scheduler de difusão e decoder), métodos de treino, imputação e forecast.| 
| `tsdiffusion.py` | Modelo de difusão latente que reutiliza o encoder Jump-ODE para denoising e imputação consistentes. |
|`ode_jump.py`, `ode_jump_encoder.py`, `ode_jump_encoder_diffusion.py`| variantes do encoder Jump-ODE com/sem transformer.| 
| `segmenter.py` |  segmentação não supervisionada (HMM) e extração de atributos.| 
|`dataloader.py` | busca dados no Cognite Data Fusion, normaliza e adiciona colunas de estado/anomalia.| 
| `gru.py`, `lstm.py`  | Baselines| 
| `deep.py`, `train.py`, `ode_jump_encoder_test.py`| Utilidades/treino|


## Dados (Cognite)
-  O DataLoader usa OAuth (Azure AD) para o projeto publicdata.
- COGNITE_CLIENT_SECRET deve estar no ambiente.
- CLIENT_ID/TENANT/PROJECT/BASE_URL já estão configurados no código.
- Por padrão busca o asset 23-KA-9101 e todas as séries associadas.
- Colunas vazias são removidas; NaN são preenchidos (ffill/bfill) e escalonados com RobustScaler.
- Métodos auxiliares:
    - add_segments(k_states, window, step, ...): roda UnsupervisedStateSegmenter, adiciona states ao DataFrame.
    - add_time_to_change_state_timestamp(): cria colunas state-<id> com timestamp da próxima mudança de estado.
    - add_next_anomaly_timestamp(column, value, boundary): cria coluna anomaly com timestamp do próximo evento acima/abaixo do limite.

## Formato esperado (DataFrame)

- feature_cols: colunas numéricas a modelar (in_channels no construtor).
- static_features_cols (opcional): atributos estáticos (static_dim).
- predict_state_cols: colunas de horizonte/estado futuro para a perda L3.
- timestamp_col: coluna temporal; use "index" para trabalhar pelo índice.
- Timestamps são normalizados por janela: (ts - t0) / TS_SPAN (TS_SPAN = 1 ano em segundos).
- Máscara de observação é derivada de NaN; os valores são zerados no forward e a máscara controla a perda.
- Perdas (lam = [L1, L2, L3, L4])
    - L1: recon com log-likelihood Gaussiano ponderado por λ(t).
    - L2: MSE do ruído previsto.
    - L3: erro no horizonte/estado previsto (predict_state_cols).
    - L4: Bernoulli para presença de observação (máscara).


## Exemplo mínimo (dados Cognite já em DataFrame)
```python
import torch
from tsdiffusion_transformed.ts_diffusion_transformed import TSDiffusion
from tsdiffusion_transformed.dataloader import DataLoader

# 1) Carrega dados do Cognite
dl = DataLoader(start=..., end=..., granularity="2s", asset_name="23-KA-9101")
dl.add_segments(segments=6, window=30, step=10)  # opcional
dl.add_time_to_change_state_timestamp()          # opcional
df = dl.df

# 2) Define colunas
feature_cols = [...]          # suas colunas de série
static_cols = [...]           # se existir
predict_state_cols = [...]    # colunas de horizonte/estado
timestamp_col = "timestamp"   # ou "index" se usar o índice

# 3) Instancia o modelo
model = TSDiffusion(
    in_channels=len(feature_cols),
    status_dim=len(predict_state_cols),
    static_dim=len(static_cols),
    hidden_dim=256,
    num_steps=1000,
    n_heads=4,
    n_layers=4,
)

# 4) Treino simples
train_loss, val_loss = model.train_model(
    df_train=df_train,
    df_val=df_val,                  # ou None
    feature_cols=feature_cols,
    static_features_cols=static_cols,
    predict_state_cols=predict_state_cols,
    timestamp_col=timestamp_col,
    status_pred_window=600 / (60*60*24*365),  # horizonte (segundos -> normalizado)
    batch_size=32,
    lr=1e-3,
    window_size=600,                # passos por janela
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# 5) Imputação
x_hat = model.impute(
    x_obs=tensor_X,     # (B, T, C) já normalizado
    mask=tensor_mask,   # (B, T, C) 1 observado / 0 ausente
    timestamps=tensor_ts,          # (B, T) 0-1
    static_feats=tensor_static,    # (B, D) ou None
    sampling_steps=200,
)

```
## Observações
- Ajuste status_pred_window ao horizonte real em segundos (o código normaliza por TS_SPAN).
- O DataLoader exige COGNITE_CLIENT_SECRET no ambiente; revise permissões/escopos antes de executar.
- GPUs são opcionais, mas recomendadas para janelas longas e num_steps altos na difusão.




## Fluxo atual (ts_diffusion_transformed.py)
1. (Opcional) Segmentar estados com `segmenter.py` e adicionar colunas ao DataFrame.
2. Treinar `TSDiffusion` com `train_model`.
3. Avaliar com `test_model` ou `evaluate_datasets`.
4. Imputar com `impute` (ou `test_impute` para avaliação controlada).
5. Prever/continuar séries com `forecast_recursive` ou `sample_continue`.

### Exemplo: treino + avaliação
```python
train_loss, val_loss = model.train_model(
    df_train=df_train,
    df_val=df_val,                  # ou None
    feature_cols=feature_cols,
    static_features_cols=static_cols,
    predict_state_cols=predict_state_cols,
    timestamp_col=timestamp_col,
    status_pred_window=600 / (60*60*24*365),
    batch_size=32,
    lr=1e-3,
    window_size=600,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
 )

test_loss = model.test_model(
    df_test=df_test,
    feature_cols=feature_cols,
    predict_state_cols=predict_state_cols,
    static_features_cols=static_cols,
    timestamp_col=timestamp_col,
    status_pred_window=600 / (60*60*24*365),
    window_size=600,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
 )
```

#### Imputação e métricas
```python
# Imputação direta (tensors já normalizados e mascarados)
x_hat = model.impute(
    x_obs=tensor_X,    # (B, T, C) z-score
    mask=tensor_mask,  # (B, T, C) 1 observado / 0 ausente
    timestamps=tensor_ts,  # (B, T) 0-1
    static_feats=tensor_static,
    sampling_steps=200,
)
# Avaliação automática de imputação (MAE/RMSE por feature)
metrics = model.evaluate_datasets(
    window_size=600,
    batch_size=256,
    missing_frac=0.2,
    sampling_steps=40,
)
```

### Forecast / continuação
```python
ts_full, x_full = model.forecast_recursive(
    x_hist=x_tensor,   # (B, L0, C) z-score
    t_hist=t_tensor,   # (B, L0) 0-1
    window_size=48,
    n_future=24,
    delta_t=1.0,       # mesmo passo temporal usado no treino
    static_feats=static_tensor,
    sampling_steps=200,
)

```