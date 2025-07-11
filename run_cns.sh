#!/bin/bash

IMAGE_NAME="bam-container-cns"
CONTAINER_NAME="bam-container-cns"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "🔧 Image '$IMAGE_NAME' not found — building it now..."
    docker build -t "$IMAGE_NAME" -f Dockerfile.cns .
else
    echo "✅ Image '$IMAGE_NAME' already exists — skipping build."
fi

# Stop and remove any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Removing existing container named $CONTAINER_NAME..."
    docker rm -f "$CONTAINER_NAME"
fi

# Run the container
docker run -it -d \
    --name "$CONTAINER_NAME" \
    --network host \
    -e DISPLAY="$DISPLAY" \
    -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --privileged \
    --runtime=nvidia \
    --gpus all \
    -v "$(pwd):/workspace" \
    -v "$HOME/.tmp:/tmp-video/" \
    "$IMAGE_NAME"

# Connect to the container
echo "🔗 Connecting to container..."
# echo "⚠️  REMEMBER TO RUN: source setup.fish INSIDE THE CONTAINER"
docker exec -it "$CONTAINER_NAME" bash