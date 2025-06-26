# Dockerfile for Very Lonely CNS - Standalone CNS Image Processor
# This creates a containerized environment for CNS image processing

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app/Very_Lonely_CNS

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/Very_Lonely_CNS:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create directories for input/output
RUN mkdir -p /app/Very_Lonely_CNS/input \
    && mkdir -p /app/Very_Lonely_CNS/results \
    && mkdir -p /app/Very_Lonely_CNS/logs

# Copy the entire Very_Lonely_CNS application
COPY . /app/Very_Lonely_CNS/

# Set proper permissions
RUN chmod +x /app/Very_Lonely_CNS/*.py

# Create a non-root user for security (optional but recommended)
RUN useradd -m -s /bin/bash cnsuser \
    && chown -R cnsuser:cnsuser /app/Very_Lonely_CNS
USER cnsuser

# Create volumes for persistent data
VOLUME ["/app/Very_Lonely_CNS/results", "/app/Very_Lonely_CNS/input"]

# Set default command to keep container running with bash
CMD ["/bin/bash"]