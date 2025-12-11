# CNN_SAE_NEU_Experiment — CNN com Sparse AutoEncoder

## Descrição
Combina um Sparse AutoEncoder (SAE) pré-treinado com uma Rede Neural Convolucional (CNN) para classificação de defetos no dataset NEU.

## Pré-requisitos
- Python 3.10+
- PyTorch
- NumPy, Pandas, Matplotlib, Scikit-learn

## Instalação
```bash
cd src/experiments/CNN_SAE_NEU_Experiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução
```bash
# Execução local
python CNN_SAE.py

# Com Singularity
singularity exec --nv --bind $PWD:/app cpe883_latest.sif python CNN_SAE.py


## Resultados
Os resultados são salvos em: **`src/Results/CNN_SAE_NEU/`**

**Pré-requisito:** Execute `SAE_NEU_Training` primeiro para gerar `encoder_best.pth`