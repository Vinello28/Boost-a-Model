FROM python:3.12-slim

# Install bash (not included by default in slim)
RUN apt-get update && \
    apt-get install -y \
    fish \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy your project files (optional)
# COPY . /workspace

# Default to bash when container starts
CMD ["fish"]
