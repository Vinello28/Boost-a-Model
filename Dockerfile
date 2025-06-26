# FROM python:3.12-slim

# # Install bash (not included by default in slim)
# RUN apt-get update && \
#     apt-get install -y \
#     fish \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     curl \
#     ffmpeg \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Support for fish plugins
# RUN fish -c "curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher"
# RUN pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 --index-url https://download.pytorch.org/whl/cu118


# # Set working directory
# WORKDIR /workspace

# CMD [ "fish" ]
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
    curl \
    ffmpeg \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Support for fish plugins
RUN fish -c "curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher"
RUN pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /workspace

CMD [ "fish" ]