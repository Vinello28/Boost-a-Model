# 🚀 ViT-VS Containerized System

Sistema Visual Servoing basato su Vision Transformer (ViT) completamente containerizzato con Docker.

## 🎯 Caratteristiche

- **Real ViT Matching**: Feature matching con DINOv2 e similarità coseno
- **GPU Accelerated**: Supporto CUDA completo
- **Memory Efficient**: Gestione intelligente della memoria GPU  
- **Production Ready**: Container ottimizzato per produzione
- **Multi-Platform**: Windows, Linux, macOS

## 📦 Contenuto Container

```
📦 ViT-VS Container
├── 🐧 Ubuntu 20.04 + CUDA 11.8
├── 🐍 Python 3.8+
├── 🧠 PyTorch + TorchVision (CUDA)
├── 🤖 DINOv2 ViT Pre-trained
├── 👁️ OpenCV + PIL
├── 📊 Matplotlib + NumPy
└── 🎯 ViT-VS System
```

## 🚀 Quick Start

### Windows
```powershell
# Build + Run
.\docker_setup.bat build
.\docker_setup.bat run

# Test sistema
.\docker_setup.bat test

# Jupyter Notebook
.\docker_setup.bat jupyter
```

### Linux/Mac
```bash
# Build + Run  
./docker_setup.sh build
./docker_setup.sh run

# Test sistema
./docker_setup.sh test

# Jupyter Notebook
./docker_setup.sh jupyter
```

### Docker Compose
```bash
# Standard
docker-compose up

# Con Jupyter
docker-compose --profile jupyter up
```

### Make (Linux/Mac)
```bash
make build    # Build immagine
make run      # Run container
make test     # Test sistema
make health   # Health check
make jupyter  # Jupyter server
```

## 📋 Prerequisiti

- **Docker Desktop** installato
- **NVIDIA Container Toolkit** (per GPU)
- **8GB RAM** disponibile
- **CUDA GPU** (opzionale, fallback su CPU)

### Installazione NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Windows:**
- Installa Docker Desktop
- Abilita WSL2 backend
- Installa NVIDIA driver

## 🧪 Test & Validation

### Health Check Completo
```bash
# Windows
docker run --rm vitqs-standalone python3 container_health_check.py

# Linux/Mac  
make health
```

**Test inclusi:**
- ✅ Importazioni Python
- ✅ GPU CUDA availability
- ✅ DINOv2 model loading
- ✅ ViT-VS system init
- ✅ Dataset access
- ✅ Output directory permissions

### Functionality Test

```bash
# Test ViT matching
docker run --rm --gpus all \
  -v $(pwd)/dataset_small:/app/dataset_small:ro \
  vitqs-standalone python3 test_vit.py
```

## 🎛️ Configurazione

### Environment Variables
```bash
# GPU control
CUDA_VISIBLE_DEVICES=0,1    # GPU da usare
NVIDIA_VISIBLE_DEVICES=all  # GPU disponibili

# Memory management
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Volume Mapping
```yaml
volumes:
  - ./dataset_small:/app/dataset_small:ro     # Dataset input
  - ./output:/app/output:rw                   # Risultati
  - ./results:/app/results:rw                 # Visualizzazioni
  - ./config:/app/config:ro                   # Configurazioni
```

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔧 Development Mode

### Interactive Shell
```bash
# Accesso completo al container
docker run -it --rm --gpus all \
  -v $(pwd):/app:rw \
  vitqs-standalone bash
```

### Jupyter Development
```bash
# Jupyter con accesso completo
make jupyter
# Apri: http://localhost:8888
```

### Live Code Editing
```bash
# Mount codice per editing live
docker run -it --rm --gpus all \
  -v $(pwd):/app:rw \
  -p 8888:8888 \
  vitqs-standalone
```

## 📊 Hardware Compatibility

### Supported Hardware
| Hardware | VRAM | Status |
|----------|------|--------|
| RTX A6000 | 48GB | Optimal - no limitations |
| RTX 4090 | 24GB | Excellent |
| RTX 3080 | 10GB | Good |
| RTX 2080 | 8GB | Standard |
| CPU Only | RAM | Functional but slower |

### Memory Requirements
- **Minimum**: 4GB RAM, 2GB VRAM
- **Recommended**: 8GB RAM, 4GB VRAM  
- **RTX A6000**: No memory constraints

## 🐛 Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Se fallisce, reinstalla nvidia-docker2
```

**2. Out of Memory**
```bash
# Forza CPU mode
docker run --rm -e CUDA_VISIBLE_DEVICES="" vitqs-standalone python3 test_vit.py

# Riduci batch size nel codice
```

**3. Permission Denied**
```bash
# Fix permessi output (Linux)
sudo chown -R $USER:$USER ./output ./results

# O usa user mapping
docker run --user $(id -u):$(id -g) ...
```

**4. Container Won't Start**
```bash
# Check logs
docker logs vitqs_container

# Debug mode
docker run -it --rm vitqs-standalone bash
```

### Debug Commands

```bash
# Sistema info
docker run --rm vitqs-standalone python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Memory check
docker run --rm --gpus all vitqs-standalone nvidia-smi

# Container size
docker images vitqs-standalone
```

## 🔐 Security

- Container esegue come utente **non-root**
- Dataset montato in **read-only**
- Porte non privilegiate
- Nessun accesso host network

## 📈 Monitoring

### Container Stats
```bash
# Real-time stats
docker stats vitqs_container

# Resource usage
docker exec vitqs_container nvidia-smi
```

### Logging
```bash
# Container logs
docker logs -f vitqs_container

# Application logs
docker exec vitqs_container tail -f /app/logs/vitqs.log
```

## 🚀 Production Deployment

### Docker Swarm
```yaml
# docker-stack.yml
version: '3.8'
services:
  vitqs:
    image: vitqs-standalone:latest
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Kubernetes
```yaml
# vitqs-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vitqs-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vitqs
  template:
    metadata:
      labels:
        app: vitqs
    spec:
      containers:
      - name: vitqs
        image: vitqs-standalone:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            memory: 4Gi
```

## 📚 Examples

### Basic Usage
```python
# In container o Jupyter
from vitqs_standalone import ViTVisualServoing

# Inizializza sistema
vitqs = ViTVisualServoing()

# Processa coppia immagini
result = vitqs.process_image_pair(
    'dataset_small/goal.jpg',
    'dataset_small/current.jpg',
    method='vit'
)

print(f"Velocity: {result['velocity']}")
```

### Batch Processing
```python
import glob

# Processa tutte le coppie
for goal_img in glob.glob('dataset_small/*.jpg'):
    for current_img in glob.glob('dataset_small/*.jpg'):
        if goal_img != current_img:
            result = vitqs.process_image_pair(goal_img, current_img)
            # Salva risultati...
```

## 🤝 Contributing

Per contribuire al container:

1. Fork del repository
2. Modifica Dockerfile o scripts
3. Test con `make health`
4. Pull request

## 📄 License

MIT License - Vedi file LICENSE per dettagli.

---

**🎉 Container ViT-VS pronto per Visual Servoing di produzione!**
