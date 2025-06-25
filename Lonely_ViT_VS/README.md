# 🤖 ViT-VS Standalone System

Sistema Visual Servoing basato su **Vision Transformer (ViT)** con **real feature matching** usando DINOv2, completamente containerizzato per deployment locale e SSH.

## 📚 **Documentazione**

- 📖 **[SERVER_README.md](SERVER_README.md)** - Documentazione completa server/client
- 📖 **[OUTPUT_FORMAT.md](OUTPUT_FORMAT.md)** - Formato dettagliato output server 
- 📖 **[INSTALLATION.md](INSTALLATION.md)** - Guida installazione step-by-step

## 🎯 Caratteristiche

- ✅ **Real ViT Matching**: Feature matching con DINOv2 e similarità del coseno
- ✅ **GPU Accelerated**: Supporto CUDA completo  
- ✅ **Multi-Environment**: Container per locale, SSH, e headless
- ✅ **Production Ready**: Docker ottimizzato per produzione

## 🚀 Installazione Rapida

### Opzione 1: Setup Automatico
```bash
python setup.py
```

### Opzione 2: Installazione Manuale
```bash
pip install -r requirements.txt
```

### Opzione 3: Test Immediato
```bash
python test_vit.py
```

## 📖 Utilizzo Base
```python
from vit_vs_standalone import ViTVisualServoing

# Inizializza il sistema ViT
vit_vs = ViTVisualServoing()

# Processa una coppia di immagini con ViT
result = vit_vs.process_image_pair(
    goal_image_path='goal.jpg',
    current_image_path='current.jpg',
    visualize=True
)

if result:
    velocity = result['velocity']  # [vx, vy, vz, ωx, ωy, ωz]
    print(f"Velocità di controllo: {velocity}")
```

### Utilizzo con Configurazione Personalizzata
```python
# Crea file di configurazione
config = {
    'u_max': 640,          # Larghezza immagine
    'v_max': 480,          # Altezza immagine
    'f_x': 554.25,         # Lunghezza focale X
    'f_y': 554.25,         # Lunghezza focale Y
    'lambda_': 0.5,        # Gain di controllo
    'num_pairs': 10,       # Numero di feature pairs
    'max_velocity': 1.0    # Velocità massima
}

import yaml
with open('my_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Usa la configurazione
vit_vs = ViTVisualServoing('my_config.yaml')
```

### Batch Processing
```python
from example_usage import batch_process_images

# Processa tutte le immagini in una directory
results = batch_process_images(
    image_dir='path/to/your/images',
    goal_image_name='goal.jpg'
)
```

## 🔧 Sistema ViT Visual Servoing

Il sistema utilizza **esclusivamente Vision Transformer (ViT)** con DINOv2 per feature extraction e matching robusto basato su deep learning.

### Caratteristiche ViT
```python
result = vit_vs.process_image_pair(goal, current, visualize=True)
```

- **Pro**: Semantic understanding, robustezza superiore
- **Feature**: Real feature matching con similarità coseno
- **Uso**: Tutte le applicazioni visual servoing
- **Performance**: Ottimizzato per GPU (RTX A6000 supporto completo)

## 📊 Output del Sistema

Il metodo `process_image_pair` ritorna un dizionario con:

```python
{
    'velocity': numpy.array,      # [vx, vy, vz, ωx, ωy, ωz]
    'goal_points': numpy.array,   # Punti nell'immagine goal
    'current_points': numpy.array, # Punti nell'immagine corrente
    'num_features': int,          # Numero di feature rilevate
    'method': str                 # Metodo utilizzato
}
```

### Interpretazione delle Velocità
- **vx, vy, vz**: Velocità lineari (m/s)
- **ωx, ωy, ωz**: Velocità angolari (rad/s)
- **Coordinate**: Camera frame (Z forward, X right, Y down)

## ⚙️ Parametri di Configurazione

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `u_max` | Larghezza immagine (pixel) | 640 |
| `v_max` | Altezza immagine (pixel) | 480 |
| `f_x`, `f_y` | Lunghezza focale | 554.25 |
| `lambda_` | Gain di controllo IBVS | 0.5 |
| `num_pairs` | Numero max di feature pairs | 10 |
| `max_velocity` | Velocità massima (m/s, rad/s) | 1.0 |

## 🧪 Testing

### Test Automatico
```bash
python example_usage.py
```

### Test con Tue Immagini
```python
from vit_vs_standalone import ViTVisualServoing

vit_vs = ViTVisualServoing()

# Test con le tue immagini
result = vit_vs.process_image_pair(
    'path/to/goal.jpg',
    'path/to/current.jpg',
    method='sift',
    visualize=True
)
```

## 📁 Struttura File

```
Lonely_ViT_VS/
├── vit_vs_standalone.py    # Sistema principale
├── simple_vit_vs.py       # Script utilizzo rapido
├── example_usage.py      # Esempi avanzati
├── setup.py              # Installazione automatica
├── requirements.txt      # Dipendenze
├── README.md             # Questa documentazione
└── test_images/          # Immagini test (auto-generate)
    ├── goal_image.jpg
    └── current_image.jpg
```

## 🔍 Algoritmo IBVS

Il sistema implementa Image-Based Visual Servoing:

1. **Feature Detection**: Rileva punti caratteristici nelle immagini
2. **Feature Matching**: Trova corrispondenze tra goal e current
3. **Error Calculation**: Calcola errore tra posizioni desiderate e attuali
4. **Interaction Matrix**: Calcola matrice di interazione L
5. **Control Law**: Applica legge di controllo `v = -λ * L^+ * e`

### Formula Matematica
```
v_c = -λ * pinv(L) * e
```
Dove:
- `v_c`: Velocità camera [vx, vy, vz, ωx, ωy, ωz]
- `λ`: Gain di controllo
- `L`: Matrice di interazione (2n×6)
- `e`: Errore features (2n×1)

## ⚠️ Caratteristiche del Sistema

1. **ViT-Based**: Sistema completamente basato su Vision Transformer
2. **DINOv2**: Utilizza modello pre-addestrato per feature extraction
3. **Real Matching**: Feature matching reale con similarità coseno
4. **GPU Accelerated**: Ottimizzato per hardware CUDA
5. **Production Ready**: Sistema robusto per applicazioni reali

## 🛠️ Estensioni Future

- [x] **Implementazione completa ViT**: Sistema completamente basato su Vision Transformer
- [x] **Feature matching robusto**: Matching bidirezionale con similarità coseno
- [x] **Ottimizzazione GPU**: Supporto completo per RTX A6000 e altre GPU
- [ ] Stima automatica profondità (monocular depth estimation)
- [ ] Calibrazione automatica camera
- [ ] Ottimizzazione per real-time processing
- [ ] Support per video streams
- [ ] Robust feature tracking

## 🐛 Troubleshooting

### Problema: "No features detected"
**Soluzione**: 
- Controlla qualità/contrasto immagini
- Prova metodi diversi (SIFT più robusto)
- Aumenta risoluzione immagini

### Problema: "Insufficient matches"
**Soluzione**:
- Riduci `num_pairs` nella configurazione
- Migliora sovrapposizione tra immagini
- Usa immagini con più texture

### Problema: "High velocities"
**Soluzione**:
- Riduci `lambda_` (gain di controllo)
- Aumenta `max_velocity` se appropriato
- Controlla errori di calibrazione camera

## 📊 Hardware Support

| Hardware | VRAM | Notes |
|----------|------|-------|
| **RTX A6000** | 48GB | No memory limitations, full resolution |
| RTX 4090 | 24GB | High capability |
| RTX 3080 | 10GB | Good capability |
| RTX 2080 | 8GB | Standard capability |
| CPU Only | RAM | Slower but functional |

**🚀 RTX A6000 Advantages:**
- 48GB VRAM - No memory sampling needed
- Full resolution ViT processing
- Maximum quality feature matching
- Supports largest ViT models

## 🚀 RTX A6000 High-Performance Setup

Per sfruttare al massimo la RTX A6000 (48GB VRAM):

### 1. Configurazione Ottimizzata
```bash
# Usa configurazione A6000
cp vit_vs_a6000_config.yaml vit_vs_config.yaml

# O imposta via Python
vit_vs = ViTVisualServoing('vit_vs_a6000_config.yaml')
```

### 2. Parametri Ottimizzati A6000

- **Model**: `dinov2_vitl14` (modello più grande)
- **Input Size**: `1024` (risoluzione più alta)  
- **Num Pairs**: `20` (più feature per accuratezza)
- **Max Patches**: Unlimited (nessun campionamento)
- **Memory Management**: Disabilitato (non necessario)

### 3. Container Setup A6000
```bash
# Build con ottimizzazioni A6000
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
./docker_setup.sh build

# Run con configurazione A6000
docker run --rm --gpus all \
  -v $(pwd)/vit_vs_a6000_config.yaml:/app/vit_vs_config.yaml \
  vit_vs-standalone:latest
```

### 4. Memory Monitoring

```bash
# Monitor GPU usage
nvidia-smi dmon -s pucvmet

# Memory usage
watch -n 1 nvidia-smi
```

## 📝 Esempi Pratici

### Dataset Processing
```python
import os
from pathlib import Path

# Processa sequenza di immagini
image_dir = Path("my_dataset")
goal_image = image_dir / "reference.jpg"

vit_vs = ViTVisualServoing()

for img_path in image_dir.glob("*.jpg"):
    if img_path != goal_image:
        result = vit_vs.process_image_pair(
            str(goal_image), 
            str(img_path)
        )
        if result:
            print(f"{img_path.name}: velocity_norm = {np.linalg.norm(result['velocity']):.3f}")
```

### Custom Visualization
```python
from vit_vs_standalone import visualize_correspondences
from PIL import Image

goal = Image.open("goal.jpg")
current = Image.open("current.jpg")

# Ottieni punti dalle tue detection
points_goal, points_current = your_detection_method(goal, current)

# Visualizza
visualize_correspondences(
    goal, current, 
    points_goal, points_current,
    save_path="my_correspondences.png"
)
```

## 💻 Requisiti Sistema

- **Python**: 3.7+
- **RAM**: 4GB+ (8GB+ raccomandati per ViT)
- **Storage**: 2GB+ per dipendenze
- **GPU**: Opzionale (CUDA per accelerazione ViT)

## 📞 Support

Per problemi o domande:
1. Controlla la sezione Troubleshooting
2. Verifica esempi in `example_usage.py`
3. Controlla configurazione parametri camera
4. Esegui `python setup.py` per test installazione

---

**🎯 Il sistema ViT-VS standalone ti permette di utilizzare l'algoritmo di visual servoing direttamente sulle tue immagini, senza la complessità del setup ROS!**
