# DNN_SAE_NEU_Experiment — DNN com Sparse AutoEncoder

## Descrição
Combina um Sparse AutoEncoder (SAE) pré-treinado com uma Rede Neural Profunda (DNN) para classificação de defetos no dataset NEU.

## Pré-requisitos
- Python 3.10+
- PyTorch
- NumPy, Pandas, Matplotlib, Scikit-learn

## Instalação
```bash
cd src/experiments/DNN_SAE_NEU_Experiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução
```bash
python DNN_SAE.py
```

# Com Singularity
singularity exec --nv --bind $PWD:/app cpe883_latest.sif python DNN_SAE.py

## Resultados
Os resultados são salvos em: **`src/Results/DNN_SAE_NEU/`**

**Pré-requisito:** Execute `SAE_NEU_Training` primeiro para gerar `encoder_best.pth`