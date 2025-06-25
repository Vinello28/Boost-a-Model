# ViT Visual Servoing Server - Output Format

Documentazione completa dell'output restituito dal server ViT Visual Servoing ai client.

## 📊 **Tipi di Response**

Il server restituisce due tipi principali di response: **Success** e **Error**.

---

## ✅ **Success Response**

Quando l'elaborazione è completata con successo, il server restituisce:

```json
{
  "status": "success",
  "velocity": [0.1234, -0.0567, 0.0890, 0.0123, -0.0456, 0.0789],
  "num_features": 15,
  "velocity_norm": 0.1654,
  "method": "vit_standalone",
  "iteration": 1,
  "processing_time": 0.234,
  "request_id": 42,
  "goal_points": [[x1, y1], [x2, y2], ...],     // Opzionale
  "current_points": [[x1, y1], [x2, y2], ...]   // Opzionale
}
```

### **Campi Success Response:**

| Campo | Tipo | Descrizione | Esempio |
|-------|------|-------------|---------|
| `status` | `string` | Stato della richiesta | `"success"` |
| `velocity` | `array[float]` | Vettore velocità IBVS [vx, vy, vz, ωx, ωy, ωz] | `[0.1234, -0.0567, 0.0890, 0.0123, -0.0456, 0.0789]` |
| `num_features` | `int` | Numero di feature corrispondenti rilevate | `15` |
| `velocity_norm` | `float` | Norma euclidea del vettore velocità | `0.1654` |
| `method` | `string` | Metodo utilizzato per il calcolo | `"vit_standalone"` |
| `iteration` | `int` | Numero iterazione del sistema | `1` |
| `processing_time` | `float` | Tempo di elaborazione in secondi | `0.234` |
| `request_id` | `int` | ID univoco della richiesta | `42` |
| `goal_points` | `array[array[float]]` | Coordinate pixel punti goal (opzionale) | `[[123.4, 567.8], [234.5, 678.9]]` |
| `current_points` | `array[array[float]]` | Coordinate pixel punti current (opzionale) | `[[125.6, 570.2], [236.7, 681.1]]` |

### **Dettagli Campi Chiave:**

#### **`velocity` - Vettore Velocità IBVS**

Array di 6 elementi rappresentante la velocità di controllo:

- `velocity[0]`: **vx** - Velocità lineare asse X (m/s)
- `velocity[1]`: **vy** - Velocità lineare asse Y (m/s)
- `velocity[2]`: **vz** - Velocità lineare asse Z (m/s)
- `velocity[3]`: **ωx** - Velocità angolare asse X (rad/s)
- `velocity[4]`: **ωy** - Velocità angolare asse Y (rad/s)
- `velocity[5]`: **ωz** - Velocità angolare asse Z (rad/s)

#### **`goal_points` / `current_points` - Coordinate Feature**

- **Formato**: Array di coordinate `[x, y]` in pixel
- **Sistema coordinate**: Origine in alto-sinistra (0,0)
- **Ordine**: Corrispondenti tra goal_points[i] e current_points[i]
- **Includere**: Solo se `include_points: true` nella richiesta

---

## ❌ **Error Response**

Quando si verifica un errore durante l'elaborazione:

```json
{
  "status": "error",
  "error": "Descrizione dettagliata dell'errore",
  "processing_time": 0.012,
  "request_id": 43
}
```

### **Campi Error Response:**

| Campo | Tipo | Descrizione | Esempio |
|-------|------|-------------|---------|
| `status` | `string` | Stato della richiesta | `"error"` |
| `error` | `string` | Messaggio di errore dettagliato | `"File non trovato: image.jpg"` |
| `processing_time` | `float` | Tempo trascorso prima dell'errore | `0.012` |
| `request_id` | `int` | ID univoco della richiesta | `43` |

### **Tipi di Errore Comuni:**

- **File non trovato**: `"File non trovato: /path/to/image.jpg"`
- **Formato immagine invalido**: `"Formato immagine non supportato"`
- **Decodifica base64 fallita**: `"Errore decodifica base64"`
- **Feature insufficienti**: `"Nessuna corrispondenza trovata"`
- **Elaborazione ViT fallita**: `"Elaborazione fallita"`
- **Parametri mancanti**: `"Dati richiesta mancanti"`

---

## 📡 **Health Check Response**

Endpoint: `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "uptime": 123.45
}
```

---

## 📊 **Stats Response**

Endpoint: `GET /stats`

```json
{
  "requests_count": 156,
  "successful_requests": 142,
  "failed_requests": 14,
  "total_processing_time": 45.67,
  "average_processing_time": 0.293,
  "uptime": 3600.12,
  "server_start_time": 1640995200.123
}
```

---

## 🔧 **System Info Response**

Endpoint: `GET /system_info`

```json
{
  "camera_params": {
    "u_max": 640,
    "v_max": 480,
    "f_x": 554.25,
    "f_y": 554.25,
    "c_x": 320.0,
    "c_y": 240.0
  },
  "control_params": {
    "lambda": 0.5,
    "max_velocity": 1.0,
    "num_pairs": 20
  },
  "vit_params": {
    "model_type": "dinov2_vits14",
    "input_size": 672,
    "device": "cuda"
  },
  "iteration_count": 5
}
```

---

## 🔌 **Socket Response Format**

Il socket server restituisce la stessa struttura JSON seguita da `\n\n`:

```json
{"status": "success", "velocity": [...], ...}\n\n
```

---

## 📝 **Esempi di Utilizzo**

### **Python Client Example**

```python
import requests

# Richiesta
response = requests.post('http://localhost:5000/process_images', json={
    'goal_image': 'goal.jpg',
    'current_image': 'current.jpg',
    'include_points': True
})

result = response.json()

if result['status'] == 'success':
    # Estrai velocità
    velocity = result['velocity']
    vx, vy, vz = velocity[0], velocity[1], velocity[2]
    wx, wy, wz = velocity[3], velocity[4], velocity[5]
    
    # Estrai punti (se richiesti)
    if 'goal_points' in result:
        goal_points = result['goal_points']
        current_points = result['current_points']
    
    # Estrai metriche
    num_features = result['num_features']
    processing_time = result['processing_time']
    
else:
    print(f"Errore: {result['error']}")
```

### **JavaScript/Node.js Example**

```javascript
const axios = require('axios');

async function processImages(goalImage, currentImage) {
    try {
        const response = await axios.post('http://localhost:5000/process_images', {
            goal_image: goalImage,
            current_image: currentImage,
            visualize: true,
            include_points: true
        });
        
        const result = response.data;
        
        if (result.status === 'success') {
            const [vx, vy, vz, wx, wy, wz] = result.velocity;
            console.log('Velocità:', { vx, vy, vz, wx, wy, wz });
            console.log('Features:', result.num_features);
            console.log('Processing time:', result.processing_time);
        } else {
            console.error('Errore:', result.error);
        }
        
    } catch (error) {
        console.error('Errore richiesta:', error.message);
    }
}
```

### **cURL Example**

```bash
# Richiesta con path file
curl -X POST http://localhost:5000/process_images \
  -H "Content-Type: application/json" \
  -d '{
    "goal_image": "goal.jpg",
    "current_image": "current.jpg",
    "include_points": true
  }' | jq .

# Output atteso:
# {
#   "status": "success",
#   "velocity": [0.1, -0.05, 0.08, 0.01, -0.04, 0.07],
#   "num_features": 15,
#   "velocity_norm": 0.165,
#   "method": "vit_standalone",
#   "iteration": 1,
#   "processing_time": 0.234,
#   "request_id": 1
# }
```

---

## ⚡ **Performance Metrics**

### **Response Times Tipici:**

- **Cold start** (primo caricamento): 2-5 secondi
- **Warm requests** (modello in VRAM): 0.1-0.3 secondi
- **Network overhead**: < 0.01 secondi (locale)

### **Dimensioni Response:**

- **Base response**: ~200-300 bytes
- **Con punti (20 feature)**: ~400-600 bytes
- **Con errore**: ~100-200 bytes

---

## 🚨 **HTTP Status Codes**

| Code | Significato | Response Type |
|------|-------------|---------------|
| `200` | Success | Success response |
| `500` | Internal Server Error | Error response |
| `404` | Endpoint non trovato | HTML error page |
| `405` | Metodo non consentito | HTML error page |

---

## 💡 **Best Practices**

1. **Controlla sempre `status`** prima di accedere agli altri campi
2. **Gestisci timeout** per richieste che potrebbero richiedere tempo
3. **Monitora `processing_time`** per ottimizzazioni
4. **Usa `include_points: false`** se non servono coordinate per ridurre bandwidth
5. **Implementa retry logic** per gestire errori temporanei
