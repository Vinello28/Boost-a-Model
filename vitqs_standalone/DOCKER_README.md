# ViT-VS Docker Container

Questo repository contiene tutto il necessario per eseguire il sistema ViT Visual Servoing in un container Docker.

## 🚀 Quick Start

### Prerequisiti
- Docker Desktop installato
- NVIDIA Container Toolkit (per GPU support)
- Almeno 8GB di RAM disponibile
- CUDA-compatible GPU (opzionale ma raccomandato)

### Costruzione e Avvio

**Windows:**
```bash
# Build dell'immagine
docker_setup.bat build

# Avvio del container
docker_setup.bat run

# Test del sistema
docker_setup.bat test

# Jupyter Notebook
docker_setup.bat jupyter
```

**Linux/Mac:**
```bash
# Rendi eseguibile lo script
chmod +x docker_setup.sh

# Build dell'immagine
./docker_setup.sh build

# Avvio del container
./docker_setup.sh run

# Test del sistema
./docker_setup.sh test

# Jupyter Notebook
./docker_setup.sh jupyter
```

### Docker Compose (Alternativo)

```bash
# Avvio standard
docker-compose up vitqs-app

# Avvio con Jupyter
docker-compose --profile jupyter up vitqs-jupyter
```

## 📦 Struttura Container

Il container include:
- **Base**: Ubuntu 20.04 + CUDA 11.8
- **Python**: 3.8+ con tutte le dipendenze
- **ViT Models**: DINOv2 pre-scaricato
- **OpenCV**: Per feature tradizionali
- **PyTorch**: Con supporto CUDA

## 🔧 Configurazione

### Volume Mounting
- `./dataset_small` → Container read-only per immagini di test
- `./output` → Container write per risultati
- `./results` → Container write per visualizzazioni

### Porte Esposte
- `8888`: Jupyter Notebook (se abilitato)

## 🎯 Comandi Disponibili

Una volta nel container:

```bash
# Test completo del sistema
python3 test_vit.py

# Demo interattivo
python3 demo.py

# Esempio d'uso
python3 example_usage.py

# Python shell per esperimenti
python3
```

## 🧪 Test del Sistema

Il container include un test automatico che verifica:
- ✅ Caricamento modelli ViT
- ✅ Estrazione feature reali
- ✅ Matching bidirezionale
- ✅ Calcolo velocità di controllo
- ✅ Confronto con metodi tradizionali

## 📊 Monitoring

### Utilizzo GPU
```bash
# Nel container
nvidia-smi

# Dall'host
docker exec vitqs_container nvidia-smi
```

### Log del Container
```bash
docker logs vitqs_container
```

## 🐛 Troubleshooting

### GPU non rilevata
```bash
# Verifica supporto NVIDIA
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Se fallisce, installa NVIDIA Container Toolkit
```

### Out of Memory
```bash
# Riduci batch size nel codice o usa CPU
docker run --rm -e CUDA_VISIBLE_DEVICES="" vitqs-standalone:latest python3 test_vit.py
```

### Permessi Volume
```bash
# Su Linux, potrebbe essere necessario
sudo chown -R $USER:$USER ./output ./results
```

## 🔧 Customizzazione

### Modifica Parametri
Edita `vitqs_config.yaml` nel volume montato:

```yaml
# Esempio configurazione
u_max: 640
v_max: 480
lambda_: 0.5
num_pairs: 10
max_velocity: 1.0
```

### Aggiunta Dataset
```bash
# Copia nuove immagini
cp your_images/* ./dataset_small/

# Riavvia container
docker restart vitqs_container
```

## 📈 Performance

### Benchmark Tipici
- **CPU (Intel i7)**: ~2-3 sec/pair
- **GPU (RTX 3080)**: ~0.5-1 sec/pair
- **Memoria**: ~2-4GB RAM, 1-2GB VRAM

### Ottimizzazione
- Usa GPU per velocità massima
- Riduci `dino_input_size` per memoria
- Aumenta `num_pairs` per accuratezza

## 🌐 Accesso Remoto

### Jupyter via Web
1. Avvia: `./docker_setup.sh jupyter`
2. Apri: http://localhost:8888
3. Token: non richiesto (development only)

### SSH nel Container
```bash
docker exec -it vitqs_container bash
```

## 📋 Variabili d'Ambiente

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU disponibili |
| `PYTHONPATH` | `/home/vitqs/vitqs_app` | Path Python |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU da usare |

## 🔐 Sicurezza

- Container esegue come utente non-root
- Volume read-only per dataset
- Nessuna porta privilegiata esposta
- Jupyter solo per development

## 📞 Support

Per problemi o domande:
1. Verifica i log: `docker logs vitqs_container`
2. Testa senza GPU: `CUDA_VISIBLE_DEVICES="" ./docker_setup.sh test`
3. Controlla spazio disco: `docker system df`

---

**🎉 Il container è pronto per Visual Servoing con ViT!**
