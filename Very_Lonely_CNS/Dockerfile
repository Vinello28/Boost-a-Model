# Base image con CUDA e cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Variabili d'ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
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
    && /opt/venv/bin/pip install --upgrade pip

# Attiva automaticamente il venv per ogni shell
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

WORKDIR /app

# Copia requirements per sfruttare la cache Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Assicurati che la directory checkpoints esista e copia i checkpoint
RUN mkdir -p /app/checkpoints
COPY checkpoints/*.pth /app/checkpoints/
COPY checkpoints/*.json /app/checkpoints/

# Copia il resto del codice
COPY . .

# Verifica che i checkpoint siano stati copiati
RUN ls -la /app/checkpoints/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

CMD ["tail", "-f", "/dev/null"]