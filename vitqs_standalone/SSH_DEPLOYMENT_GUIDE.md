# 🔐 ViT-VS SSH Deployment Guide

Guida completa per il deployment del sistema ViT-VS su server Ubuntu via SSH.

## 🎯 Panoramica

Il sistema ViT-VS può essere utilizzato su server remoti tramite SSH in due modalità:
- **X11 Forwarding**: Visualizzazione real-time sul client locale
- **Headless Mode**: Esecuzione senza display, output salvato su file

## 📋 Prerequisiti Server

### 1. Server Ubuntu Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (per GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install X11 tools
sudo apt install -y x11-apps xauth xvfb
```

### 2. Client Setup
```bash
# Windows (con X Server)
# Installa Xming o VcXsrv
# Configura DISPLAY=localhost:0.0

# Linux/Mac
# X11 già incluso
```

## 🚀 Deployment Steps

### 1. Transfer Files al Server
```bash
# Da client a server
scp -r vitqs_standalone/ user@server:/home/user/

# O clone repository
ssh user@server
git clone <repository-url> vitqs_standalone
cd vitqs_standalone
```

### 2. Build Container
```bash
# SSH nel server
ssh user@server
cd vitqs_standalone

# Build immagine Docker
./docker_setup.sh build
```

### 3. Test Environment
```bash
# Verifica SSH environment
python3 ssh_environment_check.py

# Test X11 forwarding (da client)
ssh -X user@server
cd vitqs_standalone
./docker_setup.sh test-x11
```

## 🖥️ Modalità di Utilizzo

### Modalità X11 (Con Display)

**1. Connessione SSH con X11:**
```bash
# Trusted X11 forwarding
ssh -Y user@server

# Standard X11 forwarding  
ssh -X user@server
```

**2. Verifica X11:**
```bash
echo $DISPLAY  # Deve mostrare :10.0 o simile
xclock         # Deve aprire orologio sul client
```

**3. Avvio ViT-VS:**
```bash
cd vitqs_standalone
./docker_setup.sh run
# oppure
make run
```

### Modalità Headless (Senza Display)

**1. Connessione SSH normale:**
```bash
ssh user@server
```

**2. Avvio Headless:**
```bash
cd vitqs_standalone
./docker_setup.sh headless
# oppure
make run-headless
```

**3. Risultati:**
```bash
# Plot salvati in:
ls results/
# correspondences.png
# ssh_test_plot.png
```

## 🔧 Configurazioni Avanzate

### 1. Performance Tuning
```bash
# Limita memoria GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

### 2. Batch Processing
```bash
# Script per processamento batch
docker run --rm --gpus all \
  --env MPLBACKEND=Agg \
  -v $(pwd)/dataset:/app/dataset:ro \
  -v $(pwd)/output:/app/output:rw \
  vitqs-standalone:latest \
  python3 batch_process.py
```

### 3. Jupyter Remote
```bash
# Avvia Jupyter server
./docker_setup.sh jupyter

# Tunnel SSH da client
ssh -L 8888:localhost:8888 user@server

# Apri browser: http://localhost:8888
```

## 🐛 Troubleshooting SSH

### Problem: "cannot connect to X server"

**Solution:**
```bash
# 1. Restart SSH with X11
ssh -X user@server

# 2. Check DISPLAY
echo $DISPLAY

# 3. Test X11
xclock

# 4. If fails, use headless
./docker_setup.sh headless
```

### Problem: "No DISPLAY variable"

**Solution:**
```bash
# Manual DISPLAY setup
export DISPLAY=:0

# Or force headless
make run-headless
```

### Problem: "X11 connection rejected"

**Solution:**
```bash
# On server, enable X11 forwarding
sudo nano /etc/ssh/sshd_config
# Add: X11Forwarding yes
sudo systemctl restart sshd

# Generate new xauth
xauth generate $DISPLAY
```

### Problem: Docker permission denied

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

### Problem: GPU not available

**Solution:**
```bash
# Test nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# If fails, install nvidia-container-toolkit
# Or use CPU mode
export CUDA_VISIBLE_DEVICES=""
```

## 📊 Performance Monitoring

### 1. System Resources
```bash
# GPU usage
nvidia-smi

# Container stats
docker stats vitqs_container

# Memory usage
free -h
df -h
```

### 2. Network Performance
```bash
# SSH connection speed
iperf3 -c server_ip

# X11 compression (for slow connections)
ssh -X -C user@server  # -C enables compression
```

## 🔒 Security Considerations

### 1. SSH Hardening
```bash
# Change default SSH port
sudo nano /etc/ssh/sshd_config
# Port 2222

# Disable root login
# PermitRootLogin no

# Key-based auth only
# PasswordAuthentication no
```

### 2. Container Security
```bash
# Run as non-root user (already configured)
# Limit container resources
docker run --memory=8g --cpus=4 ...

# Use read-only volumes where possible
-v dataset:ro
```

### 3. Firewall Setup
```bash
# UFW basic setup
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8888  # for Jupyter
```

## 🚀 Production Deployment

### 1. Service Setup
```bash
# Systemd service for ViT-VS
sudo nano /etc/systemd/system/vitqs.service
```

```ini
[Unit]
Description=ViT-VS Container Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=true
WorkingDirectory=/home/user/vitqs_standalone
ExecStart=/usr/bin/docker-compose up -d vitqs-headless
ExecStop=/usr/bin/docker-compose down
User=user

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
sudo systemctl enable vitqs.service
sudo systemctl start vitqs.service
```

### 2. Monitoring Setup
```bash
# Install monitoring
sudo apt install htop iotop

# Docker logging
docker logs -f vitqs_container
```

### 3. Backup Strategy
```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_${DATE}.tar.gz \
  vitqs_standalone/ \
  output/ \
  results/

# Cron backup (daily at 2 AM)
crontab -e
# 0 2 * * * /home/user/backup_vitqs.sh
```

## 📞 Support Commands

```bash
# Quick health check
make ssh-check

# Full system test
make health

# X11 test
make test-x11

# Container logs
docker logs vitqs_container

# Resource usage
docker exec vitqs_container nvidia-smi
```

---

**🎉 Il sistema ViT-VS è ora pronto per deployment SSH!**
