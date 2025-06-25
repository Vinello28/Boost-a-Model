#!/usr/bin/env python3
"""
Avvio rapido del server ViT Visual Servoing
Script semplificato per avviare il server in VRAM
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Verifica che i requirements siano installati"""
    try:
        import torch
        import flask
        import requests
        from PIL import Image
        import numpy as np
        return True
    except ImportError as e:
        print(f"❌ Dipendenza mancante: {e}")
        print("💡 Installa i requirements con: pip install -r requirements.txt")
        return False

def main():
    """Avvia il server"""
    print("🚀 ViT Visual Servoing - Quick Start")
    print("=" * 50)
    
    # Verifica requirements
    if not check_requirements():
        return
    
    # Verifica configurazione
    config_file = "vitvs_config.yaml"
    if Path(config_file).exists():
        print(f"📝 Configurazione trovata: {config_file}")
    else:
        print("⚠️  File configurazione non trovato, usando parametri default")
    
    # Verifica immagini test
    test_images = ["dataset_small/comandovitruviano.jpeg", "dataset_small/curr3.jpeg"]
    images_found = all(Path(img).exists() for img in test_images)
    
    if images_found:
        print("✅ Immagini di test trovate")
    else:
        print("⚠️  Immagini di test non trovate in dataset_small/")
    
    print("\n🎯 Avvio server...")
    print("📡 REST API sarà disponibile su: http://localhost:5000")
    print("🔌 Socket server sarà disponibile su: localhost:6000")
    print("\n💡 Per testare il server:")
    print("   python vit_vs_client.py")
    print("\n🛑 Per fermare il server: Ctrl+C")
    print("=" * 50)
    
    # Avvia server
    try:
        from vit_vs_server import ViTVSServer
        
        server = ViTVSServer(
            config_path=config_file if Path(config_file).exists() else None,
            port=5000,
            socket_port=6000
        )
        
        server.start_server()
        
    except KeyboardInterrupt:
        print("\n🛑 Server fermato dall'utente")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
