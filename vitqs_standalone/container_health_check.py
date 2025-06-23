#!/usr/bin/env python3
"""
Container Health Check per ViT-VS
Verifica che tutti i componenti funzionino correttamente nel container
"""

import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test importazioni principali"""
    print("🔬 Test importazioni...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        from PIL import Image
        print("✅ PIL/Pillow")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
        
        return True
    except Exception as e:
        print(f"❌ Errore importazioni: {e}")
        return False

def test_gpu():
    """Test disponibilità GPU"""
    print("\n🎮 Test GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            print(f"✅ CUDA disponibile")
            print(f"   Dispositivi: {device_count}")
            print(f"   Dispositivo corrente: {current_device}")
            print(f"   Nome: {device_name}")
            print(f"   Memoria: {memory_gb:.1f}GB")
            
            # Test allocazione memoria con RTX A6000
            if "A6000" in device_name:
                print("🚀 RTX A6000 rilevata - Testing high memory allocation...")
                test_tensor = torch.randn(5000, 5000).cuda()  # Bigger test for A6000
                print("✅ Test allocazione GPU A6000 OK")
            else:
                test_tensor = torch.randn(1000, 1000).cuda()
                print("✅ Test allocazione GPU OK")
            del test_tensor
            torch.cuda.empty_cache()
            
            return True
        else:
            print("⚠️  CUDA non disponibile")
            return False
            
    except Exception as e:
        print(f"❌ Errore GPU: {e}")
        return False

def test_vit_model():
    """Test caricamento modello ViT"""
    print("\n🤖 Test modello ViT...")
    
    try:
        import torch
        
        # Carica modello DINOv2
        print("📥 Caricamento DINOv2...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        print("✅ Modello caricato")
        
        # Test inference
        test_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            model = model.cuda()
            test_input = test_input.cuda()
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Test inference OK - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Errore modello ViT: {e}")
        traceback.print_exc()
        return False

def test_vitqs_system():
    """Test sistema ViT-VS completo"""
    print("\n🎯 Test sistema ViT-VS...")
    
    try:
        from vitqs_standalone import ViTVisualServoing
        
        # Inizializza sistema
        vitqs = ViTVisualServoing()
        print("✅ Sistema inizializzato")
        
        # Verifica parametri
        assert vitqs.u_max == 640
        assert vitqs.v_max == 480
        assert vitqs.num_pairs == 10
        print("✅ Parametri verificati")
        
        # Test ViT extractor
        extractor = vitqs.vit_extractor
        assert extractor is not None
        print("✅ ViT extractor OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore sistema ViT-VS: {e}")
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset immagini"""
    print("\n📁 Test dataset...")
    
    try:
        dataset_path = Path("dataset_small")
        
        if not dataset_path.exists():
            print("❌ Directory dataset non trovata")
            return False
        
        # Trova immagini
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(dataset_path.glob(ext)))
        
        if len(image_files) < 2:
            print(f"❌ Immagini insufficienti: {len(image_files)} < 2")
            return False
        
        print(f"✅ Trovate {len(image_files)} immagini")
        
        # Test caricamento immagine
        from PIL import Image
        test_image = Image.open(image_files[0])
        print(f"✅ Test caricamento OK - Size: {test_image.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore dataset: {e}")
        return False

def test_output_directories():
    """Test directory di output"""
    print("\n📂 Test directory output...")
    
    try:
        # Crea directory se non esistono
        output_dirs = ['output', 'results']
        
        for dir_name in output_dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            
            # Test scrittura
            test_file = dir_path / "test.txt"
            test_file.write_text("test")
            test_file.unlink()  # Rimuovi file test
            
            print(f"✅ Directory {dir_name} OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore directory output: {e}")
        return False

def run_full_test():
    """Esegue test completo del container"""
    print("🏥 ViT-VS Container Health Check")
    print("=" * 50)
    
    tests = [
        ("Importazioni", test_imports),
        ("GPU", test_gpu),
        ("Modello ViT", test_vit_model),
        ("Sistema ViT-VS", test_vitqs_system),
        ("Dataset", test_dataset),
        ("Directory Output", test_output_directories),
    ]
    
    results = []
    total_time = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            success = test_func()
            test_time = time.time() - start_time
            total_time += test_time
            
            results.append((test_name, success, test_time))
            
            if success:
                print(f"✅ {test_name} PASSED ({test_time:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            test_time = time.time() - start_time
            total_time += test_time
            results.append((test_name, False, test_time))
            print(f"❌ {test_name} ERROR: {e} ({test_time:.2f}s)")
    
    # Riepilogo finale
    print(f"\n{'='*60}")
    print("📊 RIEPILOGO HEALTH CHECK")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, test_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name:<20} ({test_time:.2f}s)")
    
    print(f"\n🎯 Risultato: {passed}/{total} test superati")
    print(f"⏱️  Tempo totale: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 CONTAINER HEALTHY - Tutti i test superati!")
        return 0
    else:
        print(f"\n⚠️  CONTAINER ISSUES - {total-passed} test falliti")
        return 1

if __name__ == "__main__":
    exit_code = run_full_test()
    sys.exit(exit_code)
