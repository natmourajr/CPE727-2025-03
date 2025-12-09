# Comparação de Modelos de Deep Learning para Detecção Precoce de Anomalias em Poços de Petróleo

Este repositório contém o código e os experimentos para comparar modelos de aprendizado profundo na detecção precoce de anomalias em poços de petróleo, utilizando o banco de dados 3W.

## Estrutura do repositório


## Requisitos

- Docker

## Como executar

1. Construa a imagem Docker
``` bash
docker build -t 3wtoolkit-pipeline .
```

2. Se necessário, baixe o dataset:
``` bash
docker run python helpers/download_dataset.py
```

## Referências
- Marins, M. A., Barros, B. D., Santos, I. H. F., & Seixas, J. M. de. (2021). *Fault detection and classification in oil wells and production/service lines using random forest*. **Journal of Petroleum Science and Engineering, 197**, 107879. https://doi.org/10.1016/j.petrol.2020.107879