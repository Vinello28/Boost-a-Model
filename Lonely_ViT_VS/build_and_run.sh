#!/bin/bash
# Script robusto per buildare, runnare e accedere al container Docker
set -e

IMAGE_NAME="lonely_vit_vs"
CONTAINER_NAME="lonely_vit_vs_container"
PORT=8080

# Build dell'immagine
echo "Building Docker image..."
if ! docker build -t $IMAGE_NAME .; then
    echo "Errore durante la build dell'immagine Docker. Controlla i log sopra."
    exit 1
fi

# Stop e rimuovi eventuale container esistente
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

echo "Eseguo container in modalità interattiva con bash..."
docker run --rm -it --name $CONTAINER_NAME --gpus "device=2" -p $PORT:$PORT -v $(pwd):/app $IMAGE_NAME bash
