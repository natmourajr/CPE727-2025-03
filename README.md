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

#### KAN: Kolmogorov-Arnold Networks
1. [KAN: Kolmogorov-Arnold Networks (artigo base)](https://arxiv.org/abs/2404.19756)
2. [A Survey on Kolmogorov-Arnold Network](https://arxiv.org/abs/2411.06078)
3. [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155)
4. [KAN-ODEs: Kolmogorov–Arnold network ordinary differential equations for learning dynamical systems and hidden physics](https://www.sciencedirect.com/science/article/pii/S0045782524006522)
5. [Kolmogorov-Arnold Graph Neural Networks](https://arxiv.org/abs/2406.18354)
6. [Kolmogorov-Arnold Networks (KANs) for Time Series Analysis](https://arxiv.org/abs/2405.08790)
7. [Kolmogorov-Arnold Networks are Radial Basis Function Networks](https://arxiv.org/abs/2405.06721)
8. [Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2409.10594)
9. [seqKAN: Sequence processing with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.14681)
10. [TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344)

#### Diffusion Probabilistic Models
1. [Diffusion Probabilistic Models (artigo base)](https://arxiv.org/abs/2006.11239)

#### Capsule Networks
1. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829v2)

#### Attention Models
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

#### Neural Operators
1. [DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193)