#!/usr/bin/env python3
"""
Client di test per ViT Visual Servoing Server
Esempi di come comunicare con il server sia via REST API che socket
"""

import requests
import json
import socket
import time
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

class ViTVSClient:
    """Client per comunicare con ViT Visual Servoing Server"""
    
    def __init__(self, rest_host='localhost', rest_port=5000, socket_host='localhost', socket_port=6000):
        self.rest_url = f"http://{rest_host}:{rest_port}"
        self.socket_host = socket_host
        self.socket_port = socket_port
    
    def encode_image_base64(self, image_path: str) -> str:
        """Codifica immagine in base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def check_server_health(self) -> dict:
        """Verifica stato server via REST"""
        try:
            response = requests.get(f"{self.rest_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_server_stats(self) -> dict:
        """Ottieni statistiche server via REST"""
        try:
            response = requests.get(f"{self.rest_url}/stats", timeout=5)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_system_info(self) -> dict:
        """Ottieni informazioni sistema via REST"""
        try:
            response = requests.get(f"{self.rest_url}/system_info", timeout=5)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def process_images_rest(self, goal_image_path: str, current_image_path: str, 
                           visualize: bool = False, save_path: str = None,
                           include_points: bool = False) -> dict:
        """
        Processa immagini via REST API
        
        Args:
            goal_image_path: Path immagine goal
            current_image_path: Path immagine current
            visualize: Se creare visualizzazione
            save_path: Path salvataggio visualizzazione
            include_points: Se includere punti nella risposta
            
        Returns:
            Dict: Risultato elaborazione
        """
        try:
            # Prepara payload
            payload = {
                'goal_image': goal_image_path,  # Usando path diretto
                'current_image': current_image_path,
                'visualize': visualize,
                'save_path': save_path,
                'include_points': include_points
            }
            
            # Invia richiesta
            response = requests.post(
                f"{self.rest_url}/process_images",
                json=payload,
                timeout=30
            )
            
            return response.json()
            
        except Exception as e:
            return {'error': str(e)}
    
    def process_images_rest_base64(self, goal_image_path: str, current_image_path: str,
                                  visualize: bool = False, save_path: str = None) -> dict:
        """Processa immagini via REST usando base64"""
        try:
            # Codifica immagini in base64
            goal_b64 = self.encode_image_base64(goal_image_path)
            current_b64 = self.encode_image_base64(current_image_path)
            
            payload = {
                'goal_image': {'base64': goal_b64},
                'current_image': {'base64': current_b64},
                'visualize': visualize,
                'save_path': save_path,
                'include_points': True
            }
            
            response = requests.post(
                f"{self.rest_url}/process_images",
                json=payload,
                timeout=30
            )
            
            return response.json()
            
        except Exception as e:
            return {'error': str(e)}
    
    def process_images_socket(self, goal_image_path: str, current_image_path: str,
                             visualize: bool = False, save_path: str = None) -> dict:
        """Processa immagini via socket"""
        try:
            # Connetti al socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.socket_host, self.socket_port))
            
            # Prepara payload
            payload = {
                'goal_image': goal_image_path,
                'current_image': current_image_path,
                'visualize': visualize,
                'save_path': save_path
            }
            
            # Invia richiesta
            request_data = json.dumps(payload) + '\n\n'
            sock.send(request_data.encode('utf-8'))
            
            # Ricevi risposta
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b'\n\n' in response_data:
                    break
            
            sock.close()
            
            # Decodifica risposta
            response = json.loads(response_data.decode('utf-8').replace('\n\n', ''))
            return response
            
        except Exception as e:
            return {'error': str(e)}
    
    def benchmark_server(self, goal_image_path: str, current_image_path: str, 
                        num_requests: int = 10) -> dict:
        """Benchmark delle prestazioni server"""
        print(f"🚀 Benchmark server: {num_requests} richieste")
        
        times_rest = []
        times_socket = []
        
        # Test REST API
        print("📡 Testing REST API...")
        for i in range(num_requests):
            start_time = time.time()
            result = self.process_images_rest(goal_image_path, current_image_path)
            end_time = time.time()
            
            if 'error' not in result:
                times_rest.append(end_time - start_time)
                print(f"   Request {i+1}/{num_requests}: {end_time - start_time:.3f}s")
            else:
                print(f"   Request {i+1}/{num_requests}: ERROR - {result['error']}")
        
        # Test Socket
        print("🔌 Testing Socket...")
        for i in range(num_requests):
            start_time = time.time()
            result = self.process_images_socket(goal_image_path, current_image_path)
            end_time = time.time()
            
            if 'error' not in result:
                times_socket.append(end_time - start_time)
                print(f"   Request {i+1}/{num_requests}: {end_time - start_time:.3f}s")
            else:
                print(f"   Request {i+1}/{num_requests}: ERROR - {result['error']}")
        
        # Risultati
        results = {
            'rest_api': {
                'successful_requests': len(times_rest),
                'average_time': sum(times_rest) / len(times_rest) if times_rest else 0,
                'min_time': min(times_rest) if times_rest else 0,
                'max_time': max(times_rest) if times_rest else 0
            },
            'socket': {
                'successful_requests': len(times_socket),
                'average_time': sum(times_socket) / len(times_socket) if times_socket else 0,
                'min_time': min(times_socket) if times_socket else 0,
                'max_time': max(times_socket) if times_socket else 0
            }
        }
        
        return results


def main():
    """Test del client"""
    print("🧪 ViT Visual Servoing Client Test")
    print("=" * 50)
    
    # Inizializza client
    client = ViTVSClient()
    
    # Verifica che il server sia in esecuzione
    print("🔍 Controllo stato server...")
    health = client.check_server_health()
    if 'error' in health:
        print(f"❌ Server non raggiungibile: {health['error']}")
        print("💡 Assicurati che il server sia avviato con: python vit_vs_server.py")
        return
    
    print(f"✅ Server online: {health}")
    
    # Ottieni informazioni sistema
    print("\n📊 Informazioni sistema:")
    info = client.get_system_info()
    if 'error' not in info:
        print(f"   Device: {info['vit_params']['device']}")
        print(f"   Model: {info['vit_params']['model_type']}")
        print(f"   Input size: {info['vit_params']['input_size']}")
    
    # Test con immagini esempio
    goal_image = "dataset_small/comandovitruviano.jpeg"
    current_image = "dataset_small/curr3.jpeg"
    
    if not (Path(goal_image).exists() and Path(current_image).exists()):
        print("⚠️  Immagini di test non trovate")
        print("💡 Assicurati che le immagini siano in dataset_small/")
        return
    
    print(f"\n🔍 Test elaborazione immagini:")
    print(f"   Goal: {goal_image}")
    print(f"   Current: {current_image}")
    
    # Test REST API
    print("\n📡 Test REST API...")
    start_time = time.time()
    result_rest = client.process_images_rest(
        goal_image, current_image,
        visualize=True,
        save_path="results/client_test_rest.png",
        include_points=True
    )
    rest_time = time.time() - start_time
    
    if 'error' not in result_rest:
        print(f"✅ REST API: {rest_time:.3f}s")
        print(f"   Features: {result_rest['num_features']}")
        print(f"   Velocity norm: {result_rest['velocity_norm']:.4f}")
    else:
        print(f"❌ REST API error: {result_rest['error']}")
    
    # Test Socket
    print("\n🔌 Test Socket...")
    start_time = time.time()
    result_socket = client.process_images_socket(
        goal_image, current_image,
        visualize=True,
        save_path="results/client_test_socket.png"
    )
    socket_time = time.time() - start_time
    
    if 'error' not in result_socket:
        print(f"✅ Socket: {socket_time:.3f}s")
        print(f"   Features: {result_socket['num_features']}")
        print(f"   Velocity norm: {result_socket['velocity_norm']:.4f}")
    else:
        print(f"❌ Socket error: {result_socket['error']}")
    
    # Test base64 (opzionale)
    print("\n📁 Test REST con base64...")
    start_time = time.time()
    result_b64 = client.process_images_rest_base64(
        goal_image, current_image,
        visualize=True,
        save_path="results/client_test_base64.png"
    )
    b64_time = time.time() - start_time
    
    if 'error' not in result_b64:
        print(f"✅ Base64: {b64_time:.3f}s")
        print(f"   Features: {result_b64['num_features']}")
    else:
        print(f"❌ Base64 error: {result_b64['error']}")
    
    # Statistiche finali server
    print("\n📊 Statistiche server:")
    stats = client.get_server_stats()
    if 'error' not in stats:
        print(f"   Richieste totali: {stats['requests_count']}")
        print(f"   Successi: {stats['successful_requests']}")
        print(f"   Errori: {stats['failed_requests']}")
        print(f"   Tempo medio: {stats['average_processing_time']:.3f}s")
        print(f"   Uptime: {stats['uptime']:.1f}s")
    
    print("\n✅ Test completato!")


if __name__ == "__main__":
    main()
