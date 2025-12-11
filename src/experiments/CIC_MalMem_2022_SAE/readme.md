# Experimento: SAE

Este diretório contém o script principal para executar o pipeline completo de treinamento e avaliação para a arquitetura **SAE** no dataset de Tuberculose.

## ⚙️ Parâmetros da Arquitetura (config.yaml)

Este modelo é configurado dinamicamente a partir do arquivo `config.yaml` localizado na raiz do projeto. Os parâmetros específicos para esta arquitetura, encontrados sob a chave `architectures:`, são:

```yaml
# ===================================================================
# 1. PARÂMETROS DE DADOS E VALIDAÇÃO
# ===================================================================
dataset:
  raw_file_name: "Obfuscated-MalMem2022.csv"
  random_seed: 117

cross_validation:
  n_splits: 10 
  test_size: 0.2
  
# ===================================================================
# 2. PARÂMETROS DE TREINAMENTO
# ===================================================================
training:
  optimizer: 'Adam'
  learning_rate: 0.00005
  weight_decay: 0.0001
  batch_size: 64
  epochs: 500
  early_stopping_patience: 25
  dropout_rate: 0.5

# ===================================================================
# 3. ARQUITETURAS DOS MODELOS
# ===================================================================
architectures:

  cnn:
    dropout_rate: 0.25
    cnn_channels: [3, 16, 32,64,128] # (Entrada, Conv1, Conv2, Conv3, Conv4)
    kernel_size: 3

  DeepNN_MLP: 
    dropout_rate: 0.5 
    hidden_layers: [512, 256, 128, 64] 

  Autoencoder_SAE: 
    pretrain_epochs: 25
    

  DeepNN_SAE_Classifier:
    dropout_rate: 0.5

  DBN_RBM: 
    pretrain_epochs: 50
    pretrain_lr: 0.001
    dropout_rate: 0.5
  
```

## 🚀 Como Executar
Este script foi projetado para ser executado a partir do diretório raiz do projeto, para que todos os imports de módulos (`modules/`, `dataloaders/`, `models/`) funcionem corretamente.

1. Verifique a Configuração:

Antes de executar, confirme se os parâmetros da arquitetura (acima) e, :

2. Execute o Script:

A partir do diretório raiz do projeto, execute o seguinte comando:

```Bash
python src/experiments/CIC_MalMem_2022_SAE/run_experiment.py
```


## 🔬 O que este script faz?
O `run_experiment.py` automatiza todo o pipeline de avaliação robusta que definimos:

Carrega as configurações do `config.yaml`.

Separa um conjunto de teste final (Hold-Out) estratificado e) do restante dos dados.

Executa uma Validação Cruzada de K-Folds (K=10) no restante dos dados (conjunto de Desenvolvimento).

Para cada fold:

Treina o modelo `SAE`.

Usa `early_stopping_patience` para salvar o melhor checkpoint com base na perda de validação.

Avalia o melhor modelo do fold no conjunto de validação com base na acurácia.

Ao final dos K-folds, ele seleciona o "modelo campeão" (o modelo do fold com a maior acurácia).

Realiza uma avaliação final, única e imparcial deste modelo campeão no conjunto Hold-Out.

## 📊 Saídas (Resultados)
Todos os artefatos deste experimento serão salvos na pasta raiz `results/` em um diretório único com timestamp, seguindo o padrão:

`results/CIC_MALMEM_2022_SAE/[YYYYMMDD_HHMMSS]/`

Este diretório conterá:

Subpastas para cada `fold_...` com logs e gráficos de perda.

A pasta `holdout_results/` com os gráficos ROC finais.

O modelo campeão salvo: `best_overall_model.pt`.

O resumo completo das métricas (com dados brutos dos folds): `summary_results.json.`