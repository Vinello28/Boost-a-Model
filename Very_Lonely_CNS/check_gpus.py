#!/usr/bin/env python3
"""
Script per controllare le GPU disponibili
"""

import torch

def check_gpus():
    print("=== GPU Information ===")
    
    # Controlla se CUDA è disponibile
    if torch.cuda.is_available():
        print(f"CUDA disponibile: Sì")
        print(f"Numero di GPU: {torch.cuda.device_count()}")
        print(f"GPU attuale: {torch.cuda.current_device()}")
        
        # Informazioni su ogni GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Nome: {props.name}")
            print(f"  Memoria totale: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Memoria libera: {torch.cuda.mem_get_info(i)[0] / 1024**3:.1f} GB")
            print(f"  Memoria usata: {torch.cuda.mem_get_info(i)[1] / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA NON disponibile")
        print("Verrà usata la CPU")
    
    print(f"\nDispositivo PyTorch di default: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

if __name__ == "__main__":
    check_gpus()
