#!/usr/bin/env python3
"""
Quick Test - Verifica rapida del sistema ViT-VS
"""

import os
import sys
import time
from pathlib import Path

def test_imports():
    """Test importazione moduli"""
    print("🧪 Testing imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🚀 CUDA disponibile: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA non disponibile")
        
        from lonely_vit_vs import ViTVisualServoing
        print("✅ ViTVisualServoing importato")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test caricamento modello"""
    print("\n🔧 Testing model loading...")
    try:
        from lonely_vit_vs import ViTVisualServoing
        
        # Usa configurazione se disponibile
        config_path = "vitvs_config.yaml" if Path("vitvs_config.yaml").exists() else None
        
        start_time = time.time()
        vit_vs = ViTVisualServoing(config_path=config_path)
        load_time = time.time() - start_time
        
        print(f"✅ Modello caricato in {load_time:.2f}s")
        
        # Info sistema
        info = vit_vs.get_system_info()
        print(f"📊 Device: {info['vit_params']['device']}")
        print(f"🔧 Model: {info['vit_params']['model_type']}")
        
        return True, vit_vs
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False, None

def test_processing():
    """Test elaborazione immagini"""
    print("\n🖼️  Testing image processing...")
    
    # Verifica immagini test
    goal_img = "dataset_small/comandovitruviano.jpeg"
    current_img = "dataset_small/curr3.jpeg"
    
    if not (Path(goal_img).exists() and Path(current_img).exists()):
        print("⚠️  Immagini test non trovate")
        return False
    
    try:
        success, vit_vs = test_model_loading()
        if not success:
            return False
        
        # Processa immagini
        start_time = time.time()
        result = vit_vs.process_image_pair(
            goal_img, current_img,
            visualize=True,
            save_path="results/quick_test.png"
        )
        process_time = time.time() - start_time
        
        if result:
            print(f"✅ Elaborazione completata in {process_time:.2f}s")
            print(f"📊 Features rilevate: {result['num_features']}")
            print(f"🎯 Velocity norm: {result['velocity_norm']:.4f}")
            print(f"💾 Output salvato in: results/quick_test.png")
            return True
        else:
            print("❌ Elaborazione fallita")
            return False
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False

def test_server_imports():
    """Test importazione moduli server"""
    print("\n🌐 Testing server imports...")
    try:
        import flask
        print(f"✅ Flask disponibile")
        
        import requests
        print(f"✅ Requests disponibile")
        
        return True
    except Exception as e:
        print(f"❌ Server import error: {e}")
        print("💡 Installa con: pip install flask requests")
        return False

def main():
    """Test principale"""
    print("🚀 ViT Visual Servoing - Quick Test")
    print("=" * 50)
    
    # Test base
    if not test_imports():
        print("\n❌ Test fallito: problemi import base")
        return
    
    # Test elaborazione
    if not test_processing():
        print("\n❌ Test fallito: problemi elaborazione immagini")
        return
    
    # Test server (opzionale)
    server_ok = test_server_imports()
    
    print("\n" + "=" * 50)
    print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("\n🎯 Sistema pronto per:")
    print("   - Elaborazione immagini standalone")
    if server_ok:
        print("   - Avvio server: python start_server.py")
    print("   - Test completo: python test_vit.py")
    print("\n💡 Per avviare il server:")
    print("   python start_server.py")

if __name__ == "__main__":
    main()
