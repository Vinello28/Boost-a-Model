# ViT Visual Servoing - Dockerfile
# Optimized container for DINOv2-based Visual Servoing System

# Use NVIDIA CUDA runtime as base image
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-pip \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create output and results directories
RUN mkdir -p output results

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for potential web interface or API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.12 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Default command - run the main ViT VS system
CMD ["python3.12", "lonely_vit_vs.py"]

# Alternative commands for different use cases:
# Test mode: docker run <image> python3.12 test_vit.py --device cuda:0
# Test on specific GPU: docker run <image> python3.12 test_vit.py --device cuda:1
# Interactive: docker run -it <image> /bin/bash
# Standalone: docker run <image> python3.12 vitqs_standalone/vitqs_standalone.py
