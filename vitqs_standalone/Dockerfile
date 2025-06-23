# ViT-VS Standalone Docker Container for SSH/X11 Environment
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Evita prompt interattivi
ENV DEBIAN_FRONTEND=noninteractive

# Installa dipendenze di sistema + X11 support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    x11-apps \
    xauth \
    xvfb \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Configura X11 forwarding e ottimizzazioni RTX A6000
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV MPLBACKEND=Qt5Agg

# Ottimizzazioni CUDA per RTX A6000 (48GB VRAM)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
ENV CUDA_VISIBLE_DEVICES=0

# Crea utente non-root
RUN useradd -m -s /bin/bash vitqs
WORKDIR /home/vitqs

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Crea directory dell'applicazione
RUN mkdir -p /home/vitqs/vitqs_app
WORKDIR /home/vitqs/vitqs_app

# Copia file dell'applicazione
COPY *.py ./
COPY dataset_small/ ./dataset_small/
COPY *.md ./

# Cambia proprietario dei file
RUN chown -R vitqs:vitqs /home/vitqs

# Switcha all'utente non-root
USER vitqs

# Pre-download dei modelli ViT
RUN python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"

# Espone porta per Jupyter (opzionale)
EXPOSE 8888

# Comando di default
CMD ["python3", "demo.py"]
