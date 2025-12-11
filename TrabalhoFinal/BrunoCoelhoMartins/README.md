README - Trabalho Final CPE727 (Deep Learning)

Este projeto implementa uma comparação entre diferentes arquiteturas de redes neurais (MLP, CNN, LSTM e GRU) para classificação multiclasse baseada em séries temporais.  

O repositório também inclui a biblioteca 3W instalada via `pip install -e .`, utilizada como toolkit modular para os modelos e treinadores.

------------------------------------------------------------
COMO EXECUTAR (LOCAL)
------------------------------------------------------------

1. Criar ambiente Python
   python3 -m venv env
   source env/bin/activate   # Linux/Mac
   env\Scripts\activate      # Windows

2. Instalar dependências
   pip install -r requirements.txt

3. Instalar biblioteca 3W localmente
   pip install -e .

4. Executar o treinamento
   python script.py

(também pode-se usar o requirements.txt com o conda)
------------------------------------------------------------
COMO EXECUTAR VIA DOCKER
------------------------------------------------------------

1. Construir a imagem:
   docker build -t finalproj .

2. Rodar o container:
   docker run --gpus all -it --rm finalproj

Se quiser mapear volumes (para salvar resultados no host):
   docker run --gpus all -it --rm -v $(pwd)/results:/app/results finalproj

------------------------------------------------------------
ESTRUTURA DO REPOSITÓRIO
------------------------------------------------------------

.
├── script.py                 # Script principal de treinamento e avaliação
├── requirements.txt          # Dependências do projeto
├── Dockerfile                # Configuração Docker
├── 3W/                       # Toolkit instalado via pip -e .
├── tests/                    # Testes feitos durante o desenvolvimento
└── README.md                 # Este arquivo



