#!/bin/bash

# Setup script for Very Lonely CNS Python environment with venv

echo "�🚀 Setting up Very Lonely CNS Python Environment with virtual environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ./input
mkdir -p ./results
mkdir -p ./logs

# Remove existing virtual environment if it exists
if [ -d "./venv" ]; then
    echo "�️ Removing existing virtual environment..."
    rm -rf ./venv
fi

# Create virtual environment
echo "🛠️ Creating Python virtual environment..."
$PYTHON_CMD -m venv ./venv

# Activate virtual environment
echo "� Activating virtual environment..."
source ./venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found. Installing basic dependencies..."
    pip install torch torchvision torchaudio opencv-python numpy scipy matplotlib tqdm scikit-image
fi

# Test installation
echo "🧪 Testing installation..."
python -c "import torch; print(f'✅ PyTorch installed: {torch.__version__}')"
python -c "import cv2; print(f'✅ OpenCV installed: {cv2.__version__}')"

# Test GPU availability
echo "🎮 Testing GPU availability..."
python -c "import torch; print(f'🎮 CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'🎮 GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Virtual environment is now active. Next steps:"
echo "1. To activate the environment in future sessions: source ./venv/bin/activate"
echo "2. Test the installation: python setup.py"
echo "3. Process images: python cns_image_processor.py --goal input/goal.jpg --current input/current.jpg"
echo ""
echo "📂 Directories created:"
echo "   - ./input/    (place your input images here)"
echo "   - ./results/  (output will be saved here)"
echo "   - ./logs/     (log files will be saved here)"
echo "   - ./venv/     (Python virtual environment)"
echo ""
echo "💡 To deactivate the virtual environment, run: deactivate"