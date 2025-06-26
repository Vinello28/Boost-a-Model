#!/bin/bash
# Script robusto per buildare, runnare e accedere al container Very Lonely CNS
set -e

IMAGE_NAME="very_lonely_cns"
CONTAINER_NAME="very_lonely_cns_container"
PORT=8000

# Funzioni di utilità
cleanup() {
    if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
        echo "🧹 Stopping and removing existing container..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker non trovato. Installa Docker prima di continuare."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon non avviato. Avvia Docker prima di continuare."
        exit 1
    fi
}

check_gpu() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "⚠️  Warning: GPU support non disponibile. Continuando senza GPU..."
        GPU_FLAG=""
    else
        echo "✅ GPU support disponibile"
        GPU_FLAG="--gpus all"
    fi
}

# Main execution
echo "🐳 Very Lonely CNS - Docker Build & Run Script"
echo "=============================================="

# Controlli preliminari
check_docker
check_gpu

# Build dell'immagine
echo "🏗️  Building Docker image '$IMAGE_NAME'..."
if ! docker build -t $IMAGE_NAME .; then
    echo "❌ Errore durante la build dell'immagine Docker. Controlla i log sopra."
    exit 1
fi

# Cleanup container esistenti
cleanup

# Crea directory locali se non esistono
mkdir -p ./input ./results ./logs

echo "🚀 Starting container in interactive mode..."
echo "   Container name: $CONTAINER_NAME"
echo "   Image: $IMAGE_NAME"
echo "   Port: $PORT"
echo "   GPU: ${GPU_FLAG:-"disabled"}"
echo ""

# Run del container in modalità interattiva
docker run --rm -it \
    --name $CONTAINER_NAME \
    $GPU_FLAG \
    -p $PORT:$PORT \
    -v "$(pwd)/input:/app/input" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd):/app" \
    $IMAGE_NAME bash

echo "👋 Container terminato. Arrivederci!"
