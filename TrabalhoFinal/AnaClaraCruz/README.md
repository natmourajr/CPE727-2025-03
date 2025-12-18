# Comparação de Modelos de Deep Learning para Detecção de Anomalias em Poços de Petróleo

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

4. Execute o modelo de interesse. Ex.:
``` bash
docker run python pipelines/run_simple_pipeline.py
```

## Referências
- Marins, M. A., Barros, B. D., Santos, I. H., Barrionuevo, D. C., Vargas, R. E. V., Prego, T. de M., de Lima, A. A., de Campos, M. L. R., da Silva, E. A. B., & Netto, S. L. (2021). Fault Detection and Classification in Oil Wells and Production/Service Lines Using Random Forest. *Journal of Petroleum Science and Engineering*, 197, 107879. https://doi.org/10.1016/j.petrol.2020.107879

- Dias, T. L. B., Marins, M. A., Pagliari, C. L., Barbosa, R. M. E., de Campos, M. L. R., da Silva, E. A. B., & Netto, S. L. (2024). Development of Oilwell Fault Classifiers Using a Wavelet-Based Multivariable Approach in a Modular Architecture. *SPE Journal*. https://doi.org/10.2118/221463-PA

- Vargas, R. E. V., Munaro, C. J., Ciarelli, P. M., Marins, M. A., Barros, B. D., Santos, I. H., de Campos, M. L. R., da Silva, E. A. B., & Netto, S. L. (2019). A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells. *Journal of Petroleum Science and Engineering*, 181, 106223. https://doi.org/10.1016/j.petrol.2019.106223

- Petrobras; UFRJ. *ThreeWToolkit*. Disponível em: <https://github.com/petrobras/3W>. Acessado em: 10 dez. 2025.

- Lea, C., Vidal, R., Reiter, A., & Hager, G. D. (2016). Temporal Convolutional Networks: A Unified Approach to Action Segmentation. In G. Hua & H. Jégou (Eds.), *Computer Vision – ECCV 2016 Workshops* (pp. 47–54). Cham: Springer International Publishing. ISBN 978-3-319-49409-8.

- O’Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. *arXiv preprint* arXiv:1511.08458.
