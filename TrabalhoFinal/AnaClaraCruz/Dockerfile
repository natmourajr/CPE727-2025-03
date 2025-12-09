FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho do projeto
WORKDIR /app
COPY . /app

# Atualiza pip e instala dependências Python
RUN pip install --upgrade pip && \
    pip install \
      numpy \
      pandas \
      matplotlib \
      scikit-learn \
      pyarrow \
      fastparquet \
      tqdm \
      jinja2 \
      pywavelets \
      jupyter

# Instala o 3WToolkit
RUN pip install --quiet git+https://github.com/petrobras/3W.git

# Comando para abrir um shell interativo
CMD ["/bin/bash"]
