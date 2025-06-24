# # Use official Python image
# FROM python:3.12

# # Set working directory
# WORKDIR /app

# # Copy requirements and install dependencies
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the code
# COPY . .

# # Expose port (modifica se necessario)
# EXPOSE 8080

# ViT Visual Servoing - Dockerfile
# Optimized container for DINOv2-based Visual Servoing System

# Base image con CUDA e cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Variabili d’ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PATH="/opt/venv/bin:$PATH"

# Installazione di Python 3.12 tramite PPA e dipendenze
RUN apt-get update && apt-get install -y \
      software-properties-common wget curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
      python3.12 python3.12-venv python3.12-dev \
      libglib2.0-0 libsm6 libxext6 libxrender-dev \
      libgomp1 libgtk-3-0 libgl1-mesa-glx libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crea un virtual environment
RUN python3.12 -m venv /opt/venv \
    && pip install --upgrade pip

WORKDIR /app

 # Copia il codice (incluso il tuo script di setup)
# COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

CMD ["tail", "-f", "/dev/null"]
