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

RUN curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish -o ~/.config/fish/functions/fisher.fish
RUN source ~/.config/fish/functions/fisher.fish && \
    fisher install jorgebucaran/fisher && \
    fisher install jorgebucaran/fish-dotenv

# Set working directory
WORKDIR /workspace

# Copy your project files (optional)
# COPY . /workspace

# Default to bash when container starts
CMD ["fish"]
