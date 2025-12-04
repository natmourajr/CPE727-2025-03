FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalação de dependências do sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Criação do ambiente virtual
RUN python3 -m venv /venv

# Adiciona o venv ao PATH para usar 'python' e 'pip' diretamente
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt /requirements.txt

# Instalação das dependências PyTorch e outras
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r /requirements.txt