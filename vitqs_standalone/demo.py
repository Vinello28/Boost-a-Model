"""
Demo Script per ViT-VS Standalone - Containerized Version
Dimostra l'utilizzo del sistema ViT Visual Servoing in Docker
"""

import sys
import os
from pathlib import Path

# Aggiungi il path del modulo
sys.path.append(str(Path(__file__).parent))

def check_environment():
    """Verifica l'ambiente di esecuzione"""
    print("🔍 Verifica ambiente...")
    
    # Check se siamo in un container
    if Path("/.dockerenv").exists():
        print("🐳 Esecuzione in container Docker")
    else:
        print("💻 Esecuzione in ambiente locale")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU disponibile: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  GPU non disponibile, usando CPU")
    except:
        print("❌ PyTorch non disponibile")
    
    # Check dataset
    dataset_path = Path("dataset_small")
    if dataset_path.exists():
        images = list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.jpg"))
        print(f"📁 Dataset trovato: {len(images)} immagini")
    else:
        print("❌ Dataset non trovato")
    
    print()

try:
    check_environment()
    from vitqs_standalone import ViTVisualServoing, create_example_config, visualize_correspondences
    print("✅ Moduli ViT-VS importati con successo")
except ImportError as e:
    print(f"❌ Errore nell'importazione: {e}")
    print("💡 Assicurati che tutti i file siano nella directory vitqs_standalone/")
    sys.exit(1)

def demo_basic_usage():
    """Demo utilizzo base del sistema"""
    print("\n🚀 Demo 1: Utilizzo Base")
    print("=" * 40)
    
    try:
        # Inizializza sistema con parametri default
        vitqs = ViTVisualServoing()
        print("✅ Sistema ViT-VS inizializzato")
        
        # Mostra parametri
        print(f"📋 Parametri caricati:")
        print(f"   Risoluzione: {vitqs.u_max}x{vitqs.v_max}")
        print(f"   Focale: fx={vitqs.f_x}, fy={vitqs.f_y}")
        print(f"   Gain: λ={vitqs.lambda_}")
        print(f"   Feature pairs: {vitqs.num_pairs}")
        
        return vitqs
        
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione: {e}")
        return None

def demo_with_config():
    """Demo con configurazione personalizzata"""
    print("\n🔧 Demo 2: Configurazione Personalizzata")
    print("=" * 50)
    
    try:
        # Crea configurazione personalizzata
        config_path = create_example_config()
        print(f"✅ Configurazione creata: {config_path}")
        
        # Inizializza con configurazione
        vitqs = ViTVisualServoing(config_path)
        print("✅ Sistema inizializzato con configurazione personalizzata")
        
        return vitqs
        
    except Exception as e:
        print(f"❌ Errore nella configurazione: {e}")
        return None

def demo_feature_methods():
    """Demo dei diversi metodi di feature detection"""
    print("\n🔍 Demo 3: Metodi di Feature Detection")
    print("=" * 50)
    
    methods = ['sift', 'orb', 'akaze']
    
    for method in methods:
        print(f"\n📝 Metodo: {method.upper()}")
        
        if method == 'sift':
            print("   - Robusto a scala e rotazione")
            print("   - Computazionalmente costoso")
            print("   - Ottimo per immagini con texture")
            
        elif method == 'orb':
            print("   - Veloce e libero da brevetti")
            print("   - Meno robusto di SIFT")
            print("   - Ideale per real-time")
            
        elif method == 'akaze':
            print("   - Buon compromesso velocità/accuratezza")
            print("   - Sensibile al rumore")
            print("   - Ottimo per dettagli fini")

def demo_algorithm_explanation():
    """Demo spiegazione algoritmo IBVS"""
    print("\n🧠 Demo 4: Algoritmo IBVS Spiegato")
    print("=" * 50)
    
    print("📚 Image-Based Visual Servoing (IBVS):")
    print()
    print("1️⃣ Feature Detection:")
    print("   - Rileva punti caratteristici nelle immagini")
    print("   - Goal image (posizione desiderata)")
    print("   - Current image (posizione attuale)")
    print()
    print("2️⃣ Feature Matching:")
    print("   - Trova corrispondenze tra le feature")
    print("   - Utilizza descrittori per matching")
    print("   - Filtra matches errati")
    print()
    print("3️⃣ Error Calculation:")
    print("   - e = s - s* (differenza pixel)")
    print("   - s: feature correnti")
    print("   - s*: feature desiderate")
    print()
    print("4️⃣ Interaction Matrix:")
    print("   - L: matrice di interazione (2n×6)")
    print("   - Collega velocità feature ↔ velocità camera")
    print("   - Dipende da profondità Z")
    print()
    print("5️⃣ Control Law:")
    print("   - v = -λ * L^+ * e")
    print("   - λ: gain di controllo")
    print("   - L^+: pseudo-inversa di L")
    print("   - Risultato: [vx, vy, vz, ωx, ωy, ωz]")

def demo_practical_example():
    """Demo esempio pratico"""
    print("\n💼 Demo 5: Esempio Pratico di Utilizzo")
    print("=" * 50)
    
    print("📋 Codice per processare le tue immagini:")
    print()
    print("```python")
    print("from vitqs_standalone import ViTVisualServoing")
    print()
    print("# Inizializza sistema")
    print("vitqs = ViTVisualServoing()")
    print()
    print("# Processa coppia di immagini")
    print("result = vitqs.process_image_pair(")
    print("    goal_image_path='reference.jpg',")
    print("    current_image_path='frame001.jpg',")
    print("    method='sift',")
    print("    visualize=True")
    print(")")
    print()
    print("# Estrai risultati")
    print("if result:")
    print("    velocity = result['velocity']")
    print("    num_features = result['num_features']")
    print("    print(f'Velocità: {velocity}')")
    print("```")
    print()
    print("📁 Struttura directory consigliata:")
    print("my_project/")
    print("├── vitqs_standalone.py")
    print("├── my_script.py")
    print("└── images/")
    print("    ├── reference.jpg")
    print("    ├── frame001.jpg")
    print("    └── frame002.jpg")

def demo_command_line():
    """Demo utilizzo da command line"""
    print("\n💻 Demo 6: Utilizzo Command Line")
    print("=" * 50)
    
    print("🚀 Per utilizzo rapido:")
    print("python simple_vitqs.py goal.jpg current.jpg")
    print()
    print("📊 Output esempio:")
    print("🤖 ViT-VS Quick Start")
    print("🎯 Goal Image: goal.jpg")
    print("📸 Current Image: current.jpg")
    print("🔍 Rilevamento feature...")
    print("   Tentativo con SIFT... ✅ 15 features")
    print("🎉 Successo con SIFT!")
    print("🎯 Velocità di controllo calcolate:")
    print("   Traslazione (m/s):")
    print("      vx = +0.0234")
    print("      vy = -0.0156")
    print("      vz = +0.0089")
    print("   Rotazione (rad/s):")
    print("      ωx = +0.0034")
    print("      ωy = -0.0021")
    print("      ωz = +0.0067")

def demo_tips_and_tricks():
    """Demo suggerimenti e trucchi"""
    print("\n💡 Demo 7: Suggerimenti e Trucchi")
    print("=" * 50)
    
    print("🎯 Per migliori risultati:")
    print()
    print("✅ Qualità immagini:")
    print("   - Usa immagini ad alta risoluzione")
    print("   - Evita motion blur")
    print("   - Buona illuminazione")
    print()
    print("✅ Sovrapposizione:")
    print("   - Almeno 50% di sovrapposizione")
    print("   - Mantieni oggetti in comune")
    print("   - Evita cambi drastici di viewpoint")
    print()
    print("✅ Texture:")
    print("   - Immagini con dettagli ricchi")
    print("   - Evita superfici lisce uniformi")
    print("   - Patterns distintivi aiutano")
    print()
    print("✅ Calibrazione:")
    print("   - Usa parametri camera corretti")
    print("   - fx, fy dalla calibrazione")
    print("   - Centro ottico (cx, cy)")
    print()
    print("⚠️ Problemi comuni:")
    print("   - 'No features detected' → Prova SIFT")
    print("   - 'Insufficient matches' → Migliora sovrapposizione")
    print("   - 'High velocities' → Riduci gain λ")

def main():
    """Funzione principale demo"""
    print("🎬 ViT-VS Standalone - Demo Completa")
    print("=" * 60)
    print("Questa demo mostra tutte le funzionalità del sistema")
    print("estratto dal progetto ROS originale")
    print()
    
    # Demo 1: Utilizzo base
    vitqs = demo_basic_usage()
    
    if vitqs is None:
        print("\n❌ Demo interrotta - problemi nell'inizializzazione")
        return
    
    # Demo 2: Configurazione
    demo_with_config()
    
    # Demo 3: Metodi di feature detection
    demo_feature_methods()
    
    # Demo 4: Algoritmo spiegato
    demo_algorithm_explanation()
    
    # Demo 5: Esempio pratico
    demo_practical_example()
    
    # Demo 6: Command line
    demo_command_line()
    
    # Demo 7: Tips & tricks
    demo_tips_and_tricks()
    
    # Conclusione
    print("\n🎉 Demo Completata!")
    print("=" * 40)
    print("✅ Sistema ViT-VS pronto all'uso")
    print("📚 Consulta README.md per documentazione completa")
    print("🚀 Inizia con: python simple_vitqs.py")
    print()
    print("🔗 Prossimi passi:")
    print("1. Prepara le tue immagini")
    print("2. Calibra i parametri camera")
    print("3. Testa con simple_vitqs.py")
    print("4. Integra nel tuo progetto")

if __name__ == "__main__":
    main()
