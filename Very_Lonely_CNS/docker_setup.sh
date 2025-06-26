#!/bin/bash

# Setup script for Very Lonely CNS Docker environment with GPU support

echo "🐳🚀 Setting up Very Lonely CNS Docker Environment with GPU support..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ./input
mkdir -p ./results
mkdir -p ./logs

# Make setup script executable (if it exists)
if [ -f "./container_setup.sh" ]; then
    echo "🔧 Setting up container scripts..."
    chmod +x ./container_setup.sh
fi

# Build the Docker image
echo "🏗️ Building Docker image..."
docker build -t very-lonely-cns .

# Create and start the container WITH GPU support
echo "🚀 Starting container with GPU support..."
docker run -d \
    --name very-lonely-cns \
    --gpus all \
    -v "$(pwd)/input:/app/Very_Lonely_CNS/input" \
    -v "$(pwd)/results:/app/Very_Lonely_CNS/results" \
    -v "$(pwd)/logs:/app/Very_Lonely_CNS/logs" \
    -v very-lonely-venv:/app/Very_Lonely_CNS/venv \
    -it \
    very-lonely-cns

# Wait a moment for container to start
sleep 3

echo "✅ Setup complete with GPU support!"
echo ""
echo "🎯 Next steps:"
echo "1. Enter the container: docker exec -it very-lonely-cns bash"
echo "2. Setup Python environment: ./container_setup.sh"
echo "3. Install dependencies: python setup.py"
echo "4. Test GPU: python -c 'import torch; print(torch.cuda.is_available())'"
echo "5. Process images: python cns_image_processor.py --goal input/goal.jpg --current input/current.jpg"
echo ""
echo "📂 Directories created:"
echo "   - ./input/    (place your input images here)"
echo "   - ./results/  (output will be saved here)"
echo "   - ./logs/     (log files will be saved here)"
