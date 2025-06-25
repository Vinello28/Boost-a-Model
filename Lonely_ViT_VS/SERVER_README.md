# ViT Visual Servoing Server - Modello in VRAM

Server che mantiene il modello ViT caricato permanentemente in VRAM per rispondere rapidamente alle richieste di Visual Servoing.

## Caratteristiche

- **🚀 Modello sempre in VRAM**: Il modello ViT rimane caricato in GPU per prestazioni massime
- **📡 API REST**: Interfaccia HTTP per integrazione facile
- **🔌 Socket Server**: Comunicazione diretta per prestazioni ottimali  
- **📊 Monitoraggio**: Statistiche in tempo reale e health check
- **🎯 Multi-format**: Supporta path file e immagini base64
- **⚡ Thread-safe**: Gestisce richieste multiple contemporaneamente

## Installazione Dipendenze

```bash
pip install -r requirements.txt
```

## Avvio Server

### Metodo 1: Script Semplificato
```bash
python start_server.py
```

### Metodo 2: Avvio Diretto
```bash
# Avvio completo (REST + Socket)
python vit_vs_server.py

# Solo REST API
python vit_vs_server.py --no-socket

# Solo Socket
python vit_vs_server.py --no-rest

# Porte personalizzate
python vit_vs_server.py --port 8080 --socket-port 8081

# Configurazione personalizzata
python vit_vs_server.py --config custom_config.yaml
```

## Uso del Server

### REST API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

**Risposta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "uptime": 123.45
}
```

#### Informazioni Sistema
```bash
curl http://localhost:5000/system_info
```

#### Statistiche Server
```bash
curl http://localhost:5000/stats
```

#### Elaborazione Immagini (Path File)
```bash
curl -X POST http://localhost:5000/process_images \
  -H "Content-Type: application/json" \
  -d '{
    "goal_image": "dataset_small/comandovitruviano.jpeg",
    "current_image": "dataset_small/curr3.jpeg",
    "visualize": true,
    "save_path": "results/output.png",
    "include_points": true
  }'
```

#### Elaborazione Immagini (Base64)
```bash
curl -X POST http://localhost:5000/process_images \
  -H "Content-Type: application/json" \
  -d '{
    "goal_image": {"base64": "iVBORw0KGgoAAAANSUhEUgAA..."},
    "current_image": {"base64": "iVBORw0KGgoAAAANSUhEUgAA..."},
    "visualize": false
  }'
```

**Risposta Tipica:**
```json
{
  "status": "success",
  "velocity": [0.1234, -0.0567, 0.0890, 0.0123, -0.0456, 0.0789],
  "num_features": 15,
  "velocity_norm": 0.1234,
  "method": "vit_standalone",
  "iteration": 1,
  "processing_time": 0.234,
  "request_id": 42,
  "goal_points": [[x1, y1], [x2, y2], ...],
  "current_points": [[x1, y1], [x2, y2], ...]
}
```

### Comunicazione Socket

Esempio Python:
```python
import socket
import json

# Connessione
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 6000))

# Richiesta
request = {
    "goal_image": "path/to/goal.jpg",
    "current_image": "path/to/current.jpg",
    "visualize": True
}
sock.send((json.dumps(request) + '\n\n').encode())

# Risposta
response = b""
while b'\n\n' not in response:
    response += sock.recv(4096)
result = json.loads(response.decode().replace('\n\n', ''))
sock.close()
```

### Client Python

Usa il client incluso per test rapidi:

```bash
# Test completo
python vit_vs_client.py

# Benchmark prestazioni
python -c "
from vit_vs_client import ViTVSClient
client = ViTVSClient()
results = client.benchmark_server('dataset_small/comandovitruviano.jpeg', 'dataset_small/curr3.jpeg', 10)
print(results)
"
```

## Parametri Richiesta

### Parametri Obbligatori
- `goal_image`: Immagine goal (path o base64)
- `current_image`: Immagine corrente (path o base64)

### Parametri Opzionali
- `visualize`: Crea visualizzazione keypoint (default: false)
- `save_path`: Path salvataggio visualizzazione
- `include_points`: Includi coordinate punti nella risposta (default: false)
- `depths`: Array profondità personalizzate per ogni punto

### Formati Immagine Supportati

1. **Path File**: `"path/to/image.jpg"`
2. **Base64**: `{"base64": "iVBORw0KGgoAAAANSUhEUgAA..."}`
3. **Path in Dict**: `{"path": "path/to/image.jpg"}`

## Monitoraggio

### Logs Server
Il server stampa informazioni dettagliate:
```
🚀 ViTExtractor using GPU: NVIDIA GeForce RTX 3080
✅ Modello ViT caricato e pronto in VRAM
🌐 REST API server avviato su http://localhost:5000
🔌 Socket server avviato su porta 6000
🔍 Processando richiesta #1
✅ Richiesta #1 completata in 0.234s
```

### Metriche Performance
- **Cold start**: ~2-5s (primo caricamento modello)
- **Warm requests**: ~0.1-0.3s (modello in VRAM)
- **Throughput**: ~3-10 req/s (dipende da GPU)
- **VRAM usage**: ~2-4GB (dipende da modello)

### Health Check Automatico
```bash
# Script monitoraggio
while true; do
  curl -s http://localhost:5000/health | jq .status
  sleep 5
done
```

## Ottimizzazione Performance

### GPU Settings
```yaml
# vitvs_config.yaml
device: 'cuda'              # Forza GPU
dino_input_size: 518         # Bilanciamento qualità/velocità
num_pairs: 15                # Meno feature = più veloce
model_type: 'dinov2_vits14'  # Modello più leggero
```

### Server Settings
- Usa thread/worker multipli per REST API
- Socket più veloce per singole connessioni
- Batch multiple immagini quando possibile

## Troubleshooting

### Server non si avvia
```bash
# Verifica dipendenze
python -c "import torch, flask, requests; print('OK')"

# Verifica GPU
python check_gpu.py

# Verifica porte
netstat -an | grep "5000\|6000"
```

### Performance basse
- Verifica device in uso: controllare logs per "GPU" vs "CPU"
- Ridurre `dino_input_size` se out of memory
- Aumentare `stride` per velocità maggiore
- Controllare driver NVIDIA aggiornati

### Memory issues
```bash
# Monitoraggio VRAM
nvidia-smi -l 1

# Pulisci cache PyTorch
python -c "import torch; torch.cuda.empty_cache()"
```

## Esempi Integrazione

### Python Flask App
```python
import requests

def get_velocity(goal_img, current_img):
    response = requests.post('http://localhost:5000/process_images', json={
        'goal_image': goal_img,
        'current_image': current_img
    })
    return response.json()['velocity']
```

### JavaScript/Node.js
```javascript
async function processImages(goalPath, currentPath) {
    const response = await fetch('http://localhost:5000/process_images', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            goal_image: goalPath,
            current_image: currentPath
        })
    });
    return await response.json();
}
```

### C++ (usando curl)
```cpp
#include <curl/curl.h>
#include <json/json.h>

// Implementazione chiamata REST API...
```

## Deployment Produzione

### Docker
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000 6000
CMD ["python", "vit_vs_server.py"]
```

### Systemd Service
```ini
[Unit]
Description=ViT Visual Servoing Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/vit-vs
ExecStart=/usr/bin/python3 vit_vs_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
