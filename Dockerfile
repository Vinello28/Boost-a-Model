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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Support for fish plugins
RUN fish -c "curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher jorgebucaran/fish-dotenv"

# Set working directory
WORKDIR /workspace

CMD [ "fish" ]
