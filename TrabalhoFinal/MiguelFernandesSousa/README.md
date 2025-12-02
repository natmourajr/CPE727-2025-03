# Comparação de Modelos de Deep Learning para Classificação Acústica Submarina utilizando o Banco de Dados IARA

**Nome:** Miguel Fernandes de Sousa  
**Email:** <miguel.sousa@coppe.ufrj.br>  
**Repositório GitHub:** <https://github.com/labsonar/IARA>
**Link para o Banco de Dados:** [https://doi.org/10.5281/zenodo.15777429](https://doi.org/10.5281/zenodo.15777429)
**Título do trabalho:** Comparação de Modelos de Deep Learning para Classificação Acústica Submarina utilizando o Banco de Dados IARA

## Resumo

Este trabalho investiga a aplicação de técnicas de aprendizado profundo para a tarefa de classificação de targets acústicos submarinos associados à assinatura acústica de embarcações, utilizando o banco de dados IARA (IARA: An Underwater Acoustic Database): dataset com 129 horas de gravações submarinas distribuídas em 1825 arquivos. O IARA faz parte do Projeto de Monitoramento Acústico da Bacia de Santos (PETROBRAS/IBAMA).

O objetivo principal é comparar o desempenho de diferentes arquiteturas de deep learning — como MLP, CNN 2D com espectrogramas e modelos temporais (CRNN ou TCN) — para classificação das categorias acústicas disponibilizadas no dataset. Serão gerados espectrogramas, aplicados filtros de seleção de trechos confiáveis (exclusion regions) recomendados pelo artigo original, e explorados procedimentos de normalização e data augmentation acústico.

## Justificativa

Modelos mais simples de baseline ainda apresentam grande margem para melhoria nas medidas de desempenho. Dessa forma,
é possível avaliar se modelos mais robustos são capazes de generalizar melhor.
