#!/bin/bash

# Cleanup script for Very Lonely CNS Docker environment

echo "🧹 Cleaning up Very Lonely CNS Docker Environment..."

# Stop and remove container
echo "🛑 Stopping and removing container..."
docker stop very-lonely-cns 2>/dev/null || echo "Container not running"
docker rm very-lonely-cns 2>/dev/null || echo "Container not found"

# Remove image (optional - uncomment if you want to remove the image too)
# echo "🗑️ Removing Docker image..."
# docker rmi very-lonely-cns 2>/dev/null || echo "Image not found"

# Remove named volume (optional - uncomment if you want to remove the venv)
# echo "📦 Removing virtual environment volume..."
# docker volume rm very-lonely-venv 2>/dev/null || echo "Volume not found"

echo "✅ Cleanup complete!"
echo ""
echo "📁 Note: Local directories (input, results, logs) are preserved"
echo "🔄 To rebuild: ./docker_setup.sh"