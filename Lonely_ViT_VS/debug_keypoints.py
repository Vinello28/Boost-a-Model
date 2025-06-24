#!/usr/bin/env python3
"""
Debug script per analizzare la qualità dei keypoints ViT
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from modules.vit_extractor import ViTExtractor
import torch

def debug_vit_keypoints(image1_path, image2_path):
    """Analizza la qualità dei keypoints ViT"""
    
    print("🔍 DEBUG: Analisi keypoints ViT")
    print("="*50)
    
    # Carica immagini
    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))
    
    if img1 is None or img2 is None:
        print("❌ Errore nel caricamento immagini")
        return
    
    print(f"📐 Dimensioni img1: {img1.shape}")
    print(f"📐 Dimensioni img2: {img2.shape}")
    
    # Inizializza ViT extractor
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 Device: {device}")
    
    vit = ViTExtractor(device=device)
    
    # Test con diverse risoluzioni e configurazioni
    configs = [
        {'size': 518, 'pairs': 10, 'name': 'Default'},
        {'size': 1024, 'pairs': 20, 'name': 'High Res'},
        {'size': 896, 'pairs': 15, 'name': 'Medium Res'},
        {'size': 518, 'pairs': 50, 'name': 'Many Pairs'}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🧪 Test: {config['name']}")
        print(f"   Risoluzione: {config['size']}px")
        print(f"   Pairs: {config['pairs']}")
        
        try:
            # Modifica temporaneamente le dimensioni target
            points1, points2 = vit.detect_vit_features(
                img1, img2, 
                num_pairs=config['pairs'],
                dino_input_size=config['size']
            )
            
            if points1 is not None and points2 is not None:
                # Calcola statistiche
                distances = np.linalg.norm(points1 - points2, axis=1)
                avg_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                # Calcola dispersione dei punti
                std_p1 = np.std(points1, axis=0)
                std_p2 = np.std(points2, axis=0)
                
                result = {
                    'config': config,
                    'num_points': len(points1),
                    'avg_distance': avg_dist,
                    'std_distance': std_dist,
                    'point_spread_1': np.mean(std_p1),
                    'point_spread_2': np.mean(std_p2),
                    'points1': points1,
                    'points2': points2
                }
                results.append(result)
                
                print(f"   ✅ {len(points1)} keypoints rilevati")
                print(f"   📏 Distanza media: {avg_dist:.2f}px")
                print(f"   📊 Dispersione punti: {np.mean(std_p1):.2f}px")
                
            else:
                print(f"   ❌ Nessun keypoint rilevato")
                
        except Exception as e:
            print(f"   ❌ Errore: {e}")
    
    # Trova la configurazione migliore
    if results:
        print(f"\n🏆 RISULTATI COMPARATIVI:")
        print("-" * 60)
        
        for r in results:
            quality_score = r['num_points'] / max(1, r['avg_distance']) * r['point_spread_1']
            print(f"{r['config']['name']:12} | "
                  f"Points: {r['num_points']:3d} | "
                  f"Dist: {r['avg_distance']:6.1f} | "
                  f"Spread: {r['point_spread_1']:6.1f} | "
                  f"Score: {quality_score:.3f}")
        
        # Visualizza il migliore
        best = max(results, key=lambda x: x['num_points'] / max(1, x['avg_distance']))
        visualize_keypoints(img1, img2, best['points1'], best['points2'], 
                          f"Best Config: {best['config']['name']}")
    
    return results

def visualize_keypoints(img1, img2, points1, points2, title):
    """Visualizza i keypoints rilevati"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Converti da BGR a RGB per matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(img1_rgb)
    ax1.scatter(points1[:, 0], points1[:, 1], c='red', s=50, alpha=0.8)
    ax1.set_title('Goal Image')
    ax1.axis('off')
    
    ax2.imshow(img2_rgb)
    ax2.scatter(points2[:, 0], points2[:, 1], c='blue', s=50, alpha=0.8)
    ax2.set_title('Current Image')
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Salva
    output_path = f'debug_keypoints_{title.replace(" ", "_").replace(":", "")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualizzazione salvata: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Usa le immagini del dataset
    dataset_path = Path("dataset_small")
    images = list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.jpg"))
    
    if len(images) >= 2:
        print(f"🖼️  Usando: {images[0].name} e {images[1].name}")
        debug_vit_keypoints(images[0], images[1])
    else:
        print("❌ Servono almeno 2 immagini in dataset_small/")
