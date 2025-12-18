FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Definir diretório de trabalho
WORKDIR /workspace

# Configurar timezone e modo não-interativo para evitar prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

# Instalar Python 3.11 e dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    tzdata \
    git \
    wget \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar Python 3.11 como padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Configurar timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Atualizar pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch com CUDA 12.8 (stable ou nightly)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Fixar NumPy para versão 1.x (compatibilidade com módulos compilados)
RUN pip install --no-cache-dir "numpy<2.0"

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto
COPY . .

# Expor porta para Jupyter
EXPOSE 8888

# Comando padrão
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
