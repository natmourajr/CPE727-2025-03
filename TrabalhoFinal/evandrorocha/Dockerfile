FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Definir diretório de trabalho
WORKDIR /workspace

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
