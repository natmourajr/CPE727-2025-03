# CPE727-2025-03
Repositório para ser utilizado durante a disciplina do PEE (Programa de Engenharia Elétrica) da Coppe/UFRJ CPE727 Aprendizado Profundo, período 2025/03. Professor Responsável: Natanael Nunes de Moura Junior

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[contributors-url]: https://github.com/natmourajr/CPE727-2025-03/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[forks-url]: https://github.com/natmourajr/CPE727-2025-03/network/members
[stars-shield]: https://img.shields.io/github/stars/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[stars-url]: https://github.com/natmourajr/CPE727-2025-03/stargazers
[issues-shield]: https://img.shields.io/github/issues/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[issues-url]: https://github.com/natmourajr/CPE727-2025-03/issues
[license-shield]: https://img.shields.io/github/license/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[license-url]: https://github.com/natmourajr/CPE727-2025-03/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/natanael-moura-junior-425a3294


## Ementa do curso

### Pipeline de Carregamento de Dados (Dataloader)

Este projeto conta com um módulo de **dataloader** desenvolvido para fornecer dados de forma eficiente, escalável e reprodutível aos modelos de aprendizado de máquina nas etapas de treinamento, validação e teste.

#### Objetivo

O objetivo do dataloader é automatizar e otimizar o processo de ingestão de dados, garantindo:
- Leitura eficiente de grandes volumes de dados
- Pré-processamento em tempo real
- Geração de lotes (batches) compatíveis com os frameworks de ML utilizados
- Controle sobre a aleatoriedade e reprodutibilidade dos experimentos
- Flexibilidade para diferentes formatos e tipos de dados (imagens, séries temporais, texto, etc.)

#### Exemplo de Uso

```python
from dataloader import CustomDataset
from torch.utils.data import DataLoader

dataset = CustomDataset(
    data_dir="dados/imagens",
    transform=transformacoes_padrao
)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```
#### Formatos de Dados Suportados
1. Imagens: .jpg, .png, .tiff
2. Dados tabulares: .csv, .xlsx, .parquet
3. Séries temporais: .npy, .hdf5
4. Dados anotados: .json, .xml


### Modelos a serem estudados

#### Baselines
1. [Árvores de Decisão](https://scikit-learn.org/stable/modules/tree.html)
2. [Máquinas de Vector Suporte](https://scikit-learn.org/stable/modules/svm.html)
3. [Multilayer Perceptron](https://pytorch.org/)

## Seminários

1. [Deep Feedforward Networks](https://github.com/natmourajr/CPE727-2025-03/tree/b90a4f003da77668480529911ec03df1c1c3891e/Seminarios/1%20-%20DeepNN)
2. [Regularization for Deep Learning](https://github.com/natmourajr/CPE727-2025-03/tree/b90a4f003da77668480529911ec03df1c1c3891e/Seminarios/2%20-%20Regularization)
3. [Optimization for Training Deep Models](https://github.com/natmourajr/CPE727-2025-03/tree/b90a4f003da77668480529911ec03df1c1c3891e/Seminarios/3%20-%20Optimization)
4. [Restricted Boltzmann Machine and Deep Belief Networks](https://github.com/natmourajr/CPE727-2025-03/tree/b90a4f003da77668480529911ec03df1c1c3891e/Seminarios/4%20-%20RBM)
5. [Convolutional Networks](https://github.com/natmourajr/CPE727-2025-03/tree/131c2798fab077d985cd4eb8965632ec57d4a12e/Seminarios/5%20-%20CNN) (Brenno Rodrigues, Gabriel Guimarães, Lucas Alexandre)
6. [Sequence Modeling: Recurrent and Recursive Nets](https://github.com/natmourajr/CPE727-2025-03/tree/131c2798fab077d985cd4eb8965632ec57d4a12e/Seminarios/6%20-%20RNN)
7. [Autoencoders and Representation Learning](https://github.com/natmourajr/CPE727-2025-03/tree/131c2798fab077d985cd4eb8965632ec57d4a12e/Seminarios/7%20-%20AE)
8. [Structured Probabilistic Models for Deep Learning and Probabilist Diffusion Models](https://github.com/natmourajr/CPE727-2025-03/tree/131c2798fab077d985cd4eb8965632ec57d4a12e/Seminarios/8%20-%20DPDM)
9. [Generative Adversarial Networks](https://github.com/natmourajr/CPE727-2025-03/tree/131c2798fab077d985cd4eb8965632ec57d4a12e/Seminarios/9%20-%20GAN)


## Estrutura de diretórios

```Bash
/
├── Semininários/             # Seminários sobre os tópicos da disciplina  
│
├── TrabalhoFinal/            # Apresentação final da disciplina
│
├── data/                     # Respectivos datasets	
│
├── src/                      # Código-fonte principal do projeto
│   ├── dataloaders/          # DataLoaders customizados
│   │
│   ├── models/               # Definições das arquiteturas PyTorch
│   │
│   ├── modules/              # Código reutilizável (treinamento, avaliação, otimização, regularização)
│   │
│   └── experiments/          # Scripts principais para rodar cada experimento
│    	  ├── Dockerfile                # Define o ambiente Python/PyTorch/CUDA
│		  │
│		  └── requirements.txt          # Lista de dependências Python
│
├── results/                  # Saídas dos experimentos (logs, checkpoints, métricas)
│
└── README.md                 # Documentação do projeto
```
