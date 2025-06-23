#!/usr/bin/env python3
"""
SSH + Docker Environment Check per ViT-VS
Verifica configurazione X11 forwarding e display
"""

import os
import sys
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def check_display_environment():
    """Verifica ambiente display"""
    print("🖥️  Display Environment Check")
    print("=" * 40)
    
    # Check variabili ambiente
    display = os.environ.get('DISPLAY')
    mpl_backend = os.environ.get('MPLBACKEND')
    
    print(f"DISPLAY: {display}")
    print(f"MPLBACKEND: {mpl_backend}")
    print(f"Current matplotlib backend: {matplotlib.get_backend()}")
    
    # Check se siamo in SSH
    ssh_client = os.environ.get('SSH_CLIENT')
    ssh_connection = os.environ.get('SSH_CONNECTION')
    
    if ssh_client or ssh_connection:
        print("🔗 SSH connection detected")
        print(f"SSH_CLIENT: {ssh_client}")
        print(f"SSH_CONNECTION: {ssh_connection}")
    else:
        print("💻 Local session")
    
    # Check X11 forwarding
    if display:
        print(f"✅ X11 Display available: {display}")
        return True
    else:
        print("❌ No X11 Display available")
        return False

def test_x11_basic():
    """Test X11 basic con xeyes o xclock"""
    print("\n🧪 Test X11 Basic Commands")
    print("=" * 30)
    
    commands = ['xeyes', 'xclock', 'xterm']
    
    for cmd in commands:
        try:
            # Test se comando esiste
            result = subprocess.run(['which', cmd], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print(f"✅ {cmd} available at: {result.stdout.strip()}")
                
                # Test launch (ma kill subito)
                try:
                    proc = subprocess.Popen([cmd], 
                                          stdout=subprocess.DEVNULL, 
                                          stderr=subprocess.DEVNULL)
                    proc.terminate()
                    print(f"🚀 {cmd} launch test: OK")
                except Exception as e:
                    print(f"❌ {cmd} launch test failed: {e}")
            else:
                print(f"❌ {cmd} not found")
                
        except Exception as e:
            print(f"❌ Error testing {cmd}: {e}")

def test_matplotlib_backends():
    """Test diversi backend matplotlib"""
    print("\n📊 Matplotlib Backend Test")
    print("=" * 30)
    
    backends = ['Agg', 'Qt5Agg', 'TkAgg']
    
    for backend in backends:
        try:
            matplotlib.use(backend)
            print(f"✅ {backend}: OK")
        except Exception as e:
            print(f"❌ {backend}: {e}")
    
    # Reset to Agg per sicurezza
    matplotlib.use('Agg')
    print(f"🔄 Reset to: {matplotlib.get_backend()}")

def test_plot_generation():
    """Test generazione plot"""
    print("\n📈 Plot Generation Test")
    print("=" * 25)
    
    try:
        # Test plot semplice
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', label='sin(x)')
        ax.set_title('Test Plot - ViT-VS SSH Environment')
        ax.set_xlabel('x')
        ax.set_ylabel('sin(x)')
        ax.legend()
        ax.grid(True)
        
        # Salva
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'ssh_test_plot.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Plot saved to: {save_path}")
        
        # Verifica file
        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"📁 File size: {size} bytes")
            return True
        else:
            print("❌ File not created")
            return False
            
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
        return False

def test_vitqs_imports():
    """Test importazioni ViTQS"""
    print("\n🤖 ViTQS Import Test")
    print("=" * 20)
    
    try:
        from vitqs_standalone import ViTVisualServoing, visualize_correspondences
        print("✅ ViTQS imports: OK")
        
        # Test inizializzazione
        vitqs = ViTVisualServoing()
        print("✅ ViTQS initialization: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ ViTQS import failed: {e}")
        return False

def generate_ssh_guide():
    """Genera guida per SSH setup"""
    print("\n📋 SSH Setup Guide")
    print("=" * 18)
    
    guide = """
Per utilizzare ViT-VS via SSH con X11 forwarding:

1. 🔐 Connessione SSH con X11:
   ssh -X username@server_ip
   # oppure
   ssh -Y username@server_ip  (trusted X11)

2. 🧪 Verifica X11:
   echo $DISPLAY
   xclock  # deve aprire un orologio

3. 🐳 Docker con X11:
   export DISPLAY=$DISPLAY
   xauth list  # verifica auth
   
4. 🚀 Avvia ViT-VS:
   ./docker_setup.sh build
   ./docker_setup.sh test-x11  # test X11
   ./docker_setup.sh run       # con X11
   # oppure
   ./docker_setup.sh headless  # senza display

5. 📁 Risultati:
   - Con X11: plot mostrati in real-time
   - Headless: plot salvati in results/
   
6. 🔧 Troubleshooting:
   - "cannot connect to X server": riavvia SSH con -X
   - "No DISPLAY variable": export DISPLAY=:0
   - Container issues: usa modalità headless
"""
    
    print(guide)
    
    # Salva guida
    with open('SSH_SETUP_GUIDE.md', 'w') as f:
        f.write("# ViT-VS SSH Setup Guide\n\n")
        f.write(guide)
    
    print("📄 Guida salvata: SSH_SETUP_GUIDE.md")

def main():
    """Main test function"""
    print("🔧 ViT-VS SSH Environment Setup Check")
    print("=" * 50)
    
    tests = [
        ("Display Environment", check_display_environment),
        ("X11 Basic", test_x11_basic),
        ("Matplotlib Backends", test_matplotlib_backends),
        ("Plot Generation", test_plot_generation),
        ("ViTQS Imports", test_vitqs_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        if success:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")
    
    total = len(results)
    print(f"\n🎯 Result: {passed}/{total} tests passed")
    
    # Raccomandazioni
    if passed < total:
        print("\n💡 Raccomandazioni:")
        if not check_display_environment():
            print("   - Riavvia SSH con: ssh -X username@server")
            print("   - Oppure usa modalità headless: ./docker_setup.sh headless")
        
    # Genera guida
    generate_ssh_guide()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
