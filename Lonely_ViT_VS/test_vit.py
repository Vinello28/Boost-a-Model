#!/usr/bin/env python3
"""
Test ViT Visual Servoing System - Standalone Version
Sistema di test per il visual servoing basato su Vision Transformer
"""

from lonely_vit_vs import ViTVisualServoing
import sys
import torch
import argparse
import os
import traceback

def test_vit_features(device=None):
    # Set device via environment variable if specified
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device.replace('cuda:', '')
        print(f"🎯 Forcing GPU device: {device}")
    
    print("🧪 Test ViT Visual Servoing System")
    print("========================================")
    print("Sistema esclusivamente basato su Vision Transformer")
    
    # Check GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory > 20:  
            print("⚡ Fantasmagorical GPU detected - No memory limitations!")
        else:
            print("📊 Standard GPU detected")
    else:
        print("⚠️  CPU mode - crazy man, no GPU no powah!")
    
    try:
        # System initialization with config if provided
        config_path = None
        if hasattr(args, 'config') and args.config:
            config_path = args.config
            print(f"📝 Using specified config: {config_path}")
        elif os.path.exists("vitvs_config.yaml"):
            config_path = "vitvs_config.yaml"
            print(f"📝 Using default config: {config_path}")
        else:
            print("📝 Using default parameters (no config file found)")
            
        vitvs = ViTVisualServoing(config_path=config_path)
        print("✅ Sistema ViT-VS inizializzato")
        
        # Show loaded configuration
        if config_path:
            print(f"📋 Parametri configurazione:")
            print(f"   Model: {vitvs.model_type}")
            print(f"   DINO input size: {vitvs.dino_input_size}")
            print(f"   Num pairs: {vitvs.num_pairs}")
            print(f"   Lambda: {vitvs.lambda_}")
            print(f"   Camera: {vitvs.u_max}x{vitvs.v_max}")
        
        # Test with images from dataset
        goal_path = "dataset_small/comandovitruviano.jpeg"
        current_path = "dataset_small/curr3.jpeg"
        
        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate save path for keypoints visualization
        from pathlib import Path
        goal_name = Path(goal_path).stem
        current_name = Path(current_path).stem
        save_path = f"{results_dir}/keypoints_{goal_name}_vs_{current_name}.png"
        
        print(f"\n🔍 Testing ViT Visual Servoing...")
        print(f"   Goal: {goal_path}")
        print(f"   Current: {current_path}")
        print(f"   Metodo: Vision Transformer (DINOv2)")
        print(f"   Output: {save_path}")
        
        # Test con ViT (sistema principale)
        result = vitvs.process_image_pair(
            goal_path, 
            current_path, 
            visualize=True,
            save_path=save_path
        )
        
        if result:
            print(f"\n✅ SUCCESS! ViT Visual Servoing funziona!")
            print(f"📊 Features rilevate: {result['num_features']}")
            print(f"🎯 Velocità calcolata:")
            velocity = result['velocity']
            print(f"   Traslazione: vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
            print(f"   Rotazione:   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}")
            
            # Calcola norma velocità per valutazione
            velocity_norm = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2 + 
                           velocity[3]**2 + velocity[4]**2 + velocity[5]**2)**0.5
            print(f"📏 Norma velocità: {velocity_norm:.4f}")
            
            # Info sulle coordinate dei punti (se disponibili)
            if 'goal_points' in result and 'current_points' in result:
                print(f"📍 Coordinate goal points: {len(result['goal_points'])} punti")
                print(f"📍 Coordinate current points: {len(result['current_points'])} punti")
                
            # Info sulla similarità media
            if 'velocity_norm' in result:
                print(f"🎯 Velocità normalizzata: {result['velocity_norm']:.4f}")
            
            # Conferma salvataggio immagine
            if os.path.exists(save_path):
                print(f"💾 Keypoints salvati in: {save_path}")
            else:
                print(f"⚠️  Attenzione: File di output non trovato in {save_path}")
                
        else:
            print("\n❌ FAILED! ViT Visual Servoing non funziona")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ViT Visual Servoing System')
    parser.add_argument('--device', type=str, default=None, 
                      help='GPU device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--config', type=str, default=None,
                      help='Config file path (optional)')
    
    args = parser.parse_args()
    
    # Set device if specified
    if args.device:
        print(f"🎯 Using specified device: {args.device}")
    
    test_vit_features(device=args.device)
