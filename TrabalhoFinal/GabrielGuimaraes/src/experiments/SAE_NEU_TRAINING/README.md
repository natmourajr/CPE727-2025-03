# SAE_NEU_Training — Sparse AutoEncoder Training

## Descrição
Script de treinamento de um Sparse AutoEncoder (SAE) no dataset NEU para aprendizado não supervisionado de representações comprimidas com esparsidade.

Arquivo do modelo: `src/models/SAE_NEU/model.py`

## Pré-requisitos
- Python 3.10+
- PyTorch
- NumPy, Pandas, Matplotlib, Scikit-learn

## Execução
```bash
# Execução local
python SAE.py

# Com Singularity
singularity exec --nv --bind $PWD:/app cpe883_latest.sif python SAE.py



## Análise
Após treinamento, o `encoder_best.pth` pode ser usado como feature extractor para:
- **CNN_SAE_NEU_Experiment** 
- **DNN_SAE_NEU_Experiment** 

