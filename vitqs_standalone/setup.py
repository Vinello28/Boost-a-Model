"""
Setup e installazione automatica per ViT-VS Standalone
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Esegui un comando e gestisci errori"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completato")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore in {description}")
        print(f"   Comando: {command}")
        print(f"   Errore: {e.stderr}")
        return False

def check_python_version():
    """Controlla versione Python"""
    print("🐍 Controllo versione Python...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Richiesto Python 3.7+")
        return False

def install_dependencies():
    """Installa dipendenze"""
    
    requirements = [
        "torch>=1.8.0",
        "torchvision>=0.9.0", 
        "timm>=0.4.12",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0"
    ]
    
    print("📦 Installazione dipendenze...")
    
    # Verifica se pip è disponibile
    if not run_command("pip --version", "Controllo pip"):
        print("❌ pip non trovato. Installa pip prima di continuare.")
        return False
    
    # Installa ogni dipendenza
    for req in requirements:
        package_name = req.split(">=")[0]
        print(f"   📦 Installando {package_name}...")
        
        if not run_command(f"pip install \"{req}\"", f"Installazione {package_name}"):
            print(f"⚠️ Errore nell'installazione di {package_name}")
            
            # Prova versione alternativa per PyTorch
            if "torch" in package_name:
                print("   🔄 Tentativo con versione CPU di PyTorch...")
                cpu_command = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
                if not run_command(cpu_command, "Installazione PyTorch CPU"):
                    return False
            else:
                return False
    
    return True

def test_installation():
    """Testa l'installazione"""
    print("\n🧪 Test dell'installazione...")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML")
    ]
    
    all_good = True
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Installazione fallita")
            all_good = False
    
    # Test GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   🚀 GPU CUDA disponibile: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   💻 Modalità CPU (GPU non disponibile)")
    except:
        pass
    
    return all_good

def create_demo_files():
    """Crea file di demo se non esistono"""
    print("\n📁 Controllo file di demo...")
    
    files_to_check = [
        "vitqs_standalone.py",
        "simple_vitqs.py", 
        "example_usage.py",
        "README.md"
    ]
    
    missing_files = []
    for file in files_to_check:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"   ✅ {file}")
    
    if missing_files:
        print(f"\n⚠️ File mancanti: {missing_files}")
        print("   Assicurati di avere tutti i file ViT-VS nella directory corrente")
        return False
    
    return True

def run_demo():
    """Esegui demo"""
    print("\n🎮 Vuoi eseguire un test di demo? (y/n): ", end="")
    
    try:
        choice = input().lower().strip()
        
        if choice == 'y' or choice == 'yes':
            print("\n🚀 Eseguendo demo...")
            
            # Crea immagini di test
            if run_command("python simple_vitqs.py", "Creazione immagini di test"):
                print("\n✅ Demo completato! Controlla i file di output.")
            else:
                print("\n❌ Demo fallito")
        else:
            print("\n⏭️ Demo saltato")
            
    except KeyboardInterrupt:
        print("\n\n⏭️ Demo saltato")

def main():
    """Funzione principale di setup"""
    
    print("🤖 ViT-VS Standalone Setup")
    print("=" * 50)
    print("Questo script configurerà automaticamente l'ambiente per ViT-VS")
    print()
    
    # Step 1: Controlla Python
    if not check_python_version():
        print("\n❌ Setup interrotto - aggiorna Python")
        return
    
    # Step 2: Controlla file
    if not create_demo_files():
        print("\n❌ Setup interrotto - file mancanti")
        return
    
    # Step 3: Installa dipendenze
    print(f"\n{'='*50}")
    print("📦 INSTALLAZIONE DIPENDENZE")
    print("Questo potrebbe richiedere alcuni minuti...")
    print(f"{'='*50}")
    
    if not install_dependencies():
        print("\n❌ Setup interrotto - errore nell'installazione")
        return
    
    # Step 4: Test installazione
    if not test_installation():
        print("\n⚠️ Alcuni moduli potrebbero non funzionare correttamente")
        print("   Riprova l'installazione o installa manualmente:")
        print("   pip install -r requirements.txt")
    
    # Step 5: Setup completato
    print(f"\n{'='*50}")
    print("🎉 SETUP COMPLETATO!")
    print(f"{'='*50}")
    
    print("\n📖 Come utilizzare ViT-VS:")
    print("   1. Utilizzo rapido:")
    print("      python simple_vitqs.py goal.jpg current.jpg")
    print("\n   2. Utilizzo avanzato:")
    print("      python example_usage.py")
    print("\n   3. Programmazione:")
    print("      from vitqs_standalone import ViTVisualServoing")
    
    print(f"\n📚 Documentazione completa: README.md")
    
    # Opzionale: Esegui demo
    run_demo()
    
    print(f"\n✅ Setup terminato! Buon lavoro con ViT-VS! 🚀")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Setup interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore inaspettato: {e}")
        print("   Riprova o installa manualmente le dipendenze")
