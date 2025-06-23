#!/usr/bin/env python3
"""
Test rapido per verificare l'implementazione ViT con RTX A6000
High-performance testing without memory limitations
"""

from vitqs_standalone import ViTVisualServoing
import sys
import torch

def test_vit_features():
    print("🧪 Test ViT Feature Matching Reale")
    print("========================================")
    
    # Check GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if "A6000" in gpu_name:
            print("⚡ RTX A6000 detected - No memory limitations!")
        else:
            print("📊 Standard GPU detected")
    else:
        print("⚠️  CPU mode - GPU recommended for optimal performance")
    
    try:
        # Inizializza sistema con parametri ottimizzati per A6000
        print("Using default parameters")
        vitqs = ViTVisualServoing()
        print("✅ Sistema ViT-VS inizializzato")
        
        # Test con immagini del dataset
        goal_path = "dataset_small/WhatsApp Image 2025-06-23 at 16.50.48.jpeg"
        current_path = "dataset_small/WhatsApp Image 2025-06-23 at 16.50.48 (1).jpeg"
        
        print(f"\n🔍 Testing ViT feature matching...")
        print(f"   Goal: {goal_path}")
        print(f"   Current: {current_path}")
        
        # Test con ViT reale
        result = vitqs.process_image_pair(
            goal_path, 
            current_path, 
            method='vit',
            visualize=True
        )
        
        if result:
            print(f"\n✅ SUCCESS! ViT matching funziona!")
            print(f"📊 Features rilevate: {result['num_features']}")
            print(f"🎯 Velocità calcolata:")
            velocity = result['velocity']
            print(f"   vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
            print(f"   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}")
        else:
            print("\n❌ FAILED! ViT matching non funziona")
            
        # Confronto con SIFT per validazione
        print(f"\n🔍 Confronto con SIFT...")
        result_sift = vitqs.process_image_pair(
            goal_path, 
            current_path, 
            method='sift',
            visualize=False
        )
        
        if result_sift:
            print(f"📊 SIFT features: {result_sift['num_features']}")
            print(f"🆚 ViT vs SIFT: {result['num_features'] if result else 0} vs {result_sift['num_features']}")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vit_features()
