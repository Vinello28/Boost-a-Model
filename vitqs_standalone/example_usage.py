"""
Esempio pratico di utilizzo del sistema ViT-VS standalone
"""

import numpy as np
from pathlib import Path
from vitqs_standalone import ViTVisualServoing, create_example_config
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def create_synthetic_test_images():
    """Crea immagini sintetiche per test"""
    print("🎨 Creando immagini di test sintetiche...")
    
    # Crea directory per le immagini
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Parametri
    width, height = 640, 480
    
    # Immagine goal (riferimento)
    goal_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Aggiungi alcuni oggetti geometrici
    cv2.rectangle(goal_img, (200, 150), (300, 250), (255, 0, 0), -1)  # Rettangolo rosso
    cv2.circle(goal_img, (400, 200), 50, (0, 255, 0), -1)  # Cerchio verde
    cv2.rectangle(goal_img, (150, 300), (250, 350), (0, 0, 255), -1)  # Rettangolo blu
    
    # Aggiungi alcuni punti di interesse
    for i in range(10):
        x, y = np.random.randint(50, width-50), np.random.randint(50, height-50)
        cv2.circle(goal_img, (x, y), 5, (0, 0, 0), -1)
    
    # Immagine current (con trasformazione)
    current_img = goal_img.copy()
    
    # Applica una piccola traslazione e rotazione
    M = cv2.getRotationMatrix2D((width//2, height//2), 5, 1.0)  # 5 gradi di rotazione
    M[0, 2] += 10  # Traslazione x
    M[1, 2] += 5   # Traslazione y
    
    current_img = cv2.warpAffine(current_img, M, (width, height))
    
    # Salva le immagini
    goal_path = test_dir / "goal_image.jpg"
    current_path = test_dir / "current_image.jpg"
    
    cv2.imwrite(str(goal_path), cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(current_path), cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Immagini salvate in: {test_dir}")
    return goal_path, current_path


def test_all_methods():
    """Testa tutti i metodi di feature detection"""
    print("\n🧪 Test di tutti i metodi di feature detection")
    print("=" * 60)
    
    # Crea immagini di test
    goal_path, current_path = create_synthetic_test_images()
    
    # Crea sistema
    vitqs = ViTVisualServoing()
    
    # Metodi da testare
    methods = ['sift', 'orb', 'akaze']  # 'vit' non ancora completamente implementato
    
    results = {}
    
    for method in methods:
        print(f"\n🔍 Testing {method.upper()}...")
        try:
            result = vitqs.process_image_pair(
                str(goal_path), 
                str(current_path), 
                method=method,
                visualize=False  # Non visualizzare per ora
            )
            
            if result:
                results[method] = result
                print(f"   ✅ Successo: {result['num_features']} feature rilevate")
                velocity = result['velocity']
                print(f"   📊 Velocità: T=[{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}], "
                      f"R=[{velocity[3]:.3f}, {velocity[4]:.3f}, {velocity[5]:.3f}]")
            else:
                print(f"   ❌ Fallimento")
                
        except Exception as e:
            print(f"   ❌ Errore: {e}")
    
    return results


def batch_process_images(image_dir, goal_image_name):
    """Processa un batch di immagini contro un'immagine goal"""
    print(f"\n📁 Processamento batch dalla directory: {image_dir}")
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ Directory non trovata: {image_dir}")
        return
    
    goal_path = image_dir / goal_image_name
    if not goal_path.exists():
        print(f"❌ Immagine goal non trovata: {goal_path}")
        return
    
    # Trova tutte le immagini nella directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    # Rimuovi l'immagine goal dalla lista
    image_files = [f for f in image_files if f.name != goal_image_name]
    
    if not image_files:
        print("❌ Nessuna immagine trovata nella directory")
        return
    
    print(f"📸 Trovate {len(image_files)} immagini da processare")
    
    # Inizializza sistema
    vitqs = ViTVisualServoing()
    
    results = []
    
    for i, current_path in enumerate(image_files[:5]):  # Limita a 5 per esempio
        print(f"\n[{i+1}/{min(5, len(image_files))}] Processando: {current_path.name}")
        
        try:
            result = vitqs.process_image_pair(
                str(goal_path),
                str(current_path),
                method='sift',
                visualize=False
            )
            
            if result:
                results.append({
                    'image': current_path.name,
                    'velocity': result['velocity'],
                    'num_features': result['num_features']
                })
                print(f"   ✅ {result['num_features']} features, velocità max: {np.max(np.abs(result['velocity'])):.3f}")
            else:
                print("   ❌ Processamento fallito")
                
        except Exception as e:
            print(f"   ❌ Errore: {e}")
    
    # Salva risultati
    if results:
        print(f"\n📊 Salvando risultati per {len(results)} immagini...")
        import json
        
        results_path = image_dir / "vitqs_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"✅ Risultati salvati in: {results_path}")
    
    return results


def analyze_convergence(results):
    """Analizza la convergenza del controllo"""
    print("\n📈 Analisi di convergenza")
    print("=" * 40)
    
    if not results:
        print("❌ Nessun risultato da analizzare")
        return
    
    velocities = [r['velocity'] for r in results]
    velocities = np.array(velocities)
    
    # Calcola norme
    translation_norms = np.linalg.norm(velocities[:, :3], axis=1)
    rotation_norms = np.linalg.norm(velocities[:, 3:], axis=1)
    
    print(f"📊 Statistiche velocità di traslazione:")
    print(f"   Media: {np.mean(translation_norms):.4f}")
    print(f"   Max: {np.max(translation_norms):.4f}")
    print(f"   Min: {np.min(translation_norms):.4f}")
    
    print(f"\n📊 Statistiche velocità di rotazione:")
    print(f"   Media: {np.mean(rotation_norms):.4f}")
    print(f"   Max: {np.max(rotation_norms):.4f}")
    print(f"   Min: {np.min(rotation_norms):.4f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(translation_norms, 'b-o', markersize=4)
    plt.title('Norma Velocità di Traslazione')
    plt.xlabel('Immagine')
    plt.ylabel('||v|| (m/s)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rotation_norms, 'r-o', markersize=4)
    plt.title('Norma Velocità di Rotazione')
    plt.xlabel('Immagine')
    plt.ylabel('||ω|| (rad/s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Grafico salvato come: convergence_analysis.png")


def main():
    """Funzione principale di esempio"""
    print("🤖 ViT-VS Standalone - Esempi di Utilizzo")
    print("=" * 60)
    
    # Test 1: Crea e testa immagini sintetiche
    print("\n1️⃣ Test con immagini sintetiche")
    results = test_all_methods()
    
    # Test 2: Esempio di utilizzo singolo
    print("\n2️⃣ Esempio di utilizzo singolo")
    goal_path, current_path = create_synthetic_test_images()
    
    vitqs = ViTVisualServoing()
    result = vitqs.process_image_pair(
        str(goal_path), 
        str(current_path), 
        method='sift',
        visualize=True,
        save_path='correspondence_example.png'
    )
    
    if result:
        print(f"🎯 Velocità calcolata con successo!")
        print(f"   Metodo: {result['method']}")
        print(f"   Feature: {result['num_features']}")
        print(f"   Velocità: {result['velocity']}")
    
    # Test 3: Istruzioni per batch processing
    print("\n3️⃣ Per processamento batch:")
    print("   - Metti le tue immagini in una directory")
    print("   - Rinomina l'immagine goal come 'goal.jpg'")
    print("   - Esegui: batch_process_images('path/to/images', 'goal.jpg')")
    
    print("\n✅ Test completati!")
    print("\n📖 Utilizzo del sistema:")
    print("from vitqs_standalone import ViTVisualServoing")
    print("vitqs = ViTVisualServoing()")
    print("result = vitqs.process_image_pair('goal.jpg', 'current.jpg', method='sift')")


if __name__ == "__main__":
    main()
