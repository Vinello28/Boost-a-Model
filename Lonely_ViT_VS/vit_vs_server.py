#!/usr/bin/env python3
"""
ViT Visual Servoing Server - Keep model in VRAM
Server che mantiene il modello ViT caricato in memoria GPU per rispondere rapidamente alle richieste
Supporta sia API REST che socket per comunicazione con client esterni
"""

import os
import sys
import json
import time
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
import socket
import argparse

# Import del sistema ViT
from lonely_vit_vs import ViTVisualServoing
from modules.utils import load_config

class ViTVSServer:
    """Server per Visual Servoing con ViT mantenuto in VRAM"""
    
    def __init__(self, config_path: Optional[str] = None, port: int = 5000, socket_port: int = 6000):
        """
        Inizializza il server ViT Visual Servoing
        
        Args:
            config_path: Path al file di configurazione
            port: Porta per API REST Flask
            socket_port: Porta per comunicazione socket
        """
        self.port = port
        self.socket_port = socket_port
        self.config_path = config_path
        
        print("🚀 Inizializzazione ViT Visual Servoing Server")
        print("=" * 60)
        
        # Verifica GPU
        self._check_gpu()
        
        # Inizializza sistema ViT (mantiene modello in VRAM)
        print("📥 Caricamento modello ViT in VRAM...")
        try:
            self.vit_vs = ViTVisualServoing(config_path=config_path)
            print("✅ Modello ViT caricato e pronto in VRAM")
            
            # Informazioni sistema
            info = self.vit_vs.get_system_info()
            print(f"📊 Device: {info['vit_params']['device']}")
            print(f"🔧 Modello: {info['vit_params']['model_type']}")
            print(f"📐 Input size: {info['vit_params']['input_size']}")
            print(f"🎯 Features: {info['control_params']['num_pairs']}")
            
        except Exception as e:
            print(f"❌ Errore caricamento modello: {e}")
            sys.exit(1)
        
        # Statistiche
        self.stats = {
            'requests_count': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'server_start_time': time.time()
        }
        
        # Flask app per API REST
        self.app = Flask(__name__)
        self._setup_flask_routes()
        
        # Socket server per comunicazione diretta
        self.socket_server = None
        self.socket_thread = None
        
        print("🎯 Server inizializzato e pronto")
        print("=" * 60)
    
    def _check_gpu(self):
        """Verifica disponibilità GPU"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU rilevata: {gpu_name}")
            print(f"💾 VRAM disponibile: {gpu_memory:.1f} GB")
        else:
            print("⚠️  GPU non disponibile - usando CPU (prestazioni ridotte)")
    
    def _setup_flask_routes(self):
        """Configura le route dell'API REST Flask"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': True,
                'device': self.vit_vs.vit_extractor.device,
                'uptime': time.time() - self.stats['server_start_time']
            })
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Statistiche server"""
            current_stats = self.stats.copy()
            current_stats['uptime'] = time.time() - self.stats['server_start_time']
            if current_stats['requests_count'] > 0:
                current_stats['average_processing_time'] = (
                    current_stats['total_processing_time'] / current_stats['requests_count']
                )
            return jsonify(current_stats)
        
        @self.app.route('/system_info', methods=['GET'])
        def get_system_info():
            """Informazioni sistema ViT"""
            return jsonify(self.vit_vs.get_system_info())
        
        @self.app.route('/process_images', methods=['POST'])
        def process_images():
            """Endpoint principale per processare immagini"""
            return self._process_images_request(request)
        
        @self.app.route('/reset_stats', methods=['POST'])
        def reset_stats():
            """Reset statistiche"""
            self.stats = {
                'requests_count': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_processing_time': 0.0,
                'average_processing_time': 0.0,
                'server_start_time': time.time()
            }
            return jsonify({'status': 'stats reset'})
    
    def _process_images_request(self, request_obj) -> Dict[str, Any]:
        """
        Processa richiesta di elaborazione immagini
        
        Args:
            request_obj: Oggetto richiesta Flask o dizionario
            
        Returns:
            Dict: Risultato elaborazione
        """
        start_time = time.time()
        self.stats['requests_count'] += 1
        
        try:
            # Estrai dati dalla richiesta
            if hasattr(request_obj, 'json'):
                data = request_obj.json
            else:
                data = request_obj
            
            # Validazione parametri
            if not data:
                raise ValueError("Dati richiesta mancanti")
            
            # Estrai immagini (base64 o path)
            goal_image = self._decode_image(data.get('goal_image'))
            current_image = self._decode_image(data.get('current_image'))
            
            if goal_image is None or current_image is None:
                raise ValueError("Immagini mancanti o non valide")
            
            # Parametri opzionali
            visualize = data.get('visualize', False)
            save_path = data.get('save_path', None)
            depths = data.get('depths', None)
            if depths:
                depths = np.array(depths)
            
            # Processa con ViT
            print(f"🔍 Processando richiesta #{self.stats['requests_count']}")
            result = self.vit_vs.process_image_pair(
                goal_image, current_image,
                depths=depths,
                visualize=visualize,
                save_path=save_path
            )
            
            if result is None:
                raise Exception("Elaborazione fallita")
            
            # Prepara risposta
            response = {
                'status': 'success',
                'velocity': result['velocity'].tolist(),
                'num_features': result['num_features'],
                'velocity_norm': float(result['velocity_norm']),
                'method': result['method'],
                'iteration': result['iteration'],
                'processing_time': time.time() - start_time,
                'request_id': self.stats['requests_count']
            }
            
            # Includi punti se richiesto
            if data.get('include_points', False):
                response['goal_points'] = result['goal_points'].tolist()
                response['current_points'] = result['current_points'].tolist()
            
            # Aggiorna statistiche
            processing_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            
            print(f"✅ Richiesta #{self.stats['requests_count']} completata in {processing_time:.3f}s")
            
            if hasattr(request_obj, 'json'):
                return jsonify(response)
            else:
                return response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_response = {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'request_id': self.stats['requests_count']
            }
            
            print(f"❌ Errore richiesta #{self.stats['requests_count']}: {e}")
            
            if hasattr(request_obj, 'json'):
                return jsonify(error_response), 500
            else:
                return error_response
    
    def _decode_image(self, image_data: Union[str, dict]) -> Optional[Image.Image]:
        """
        Decodifica immagine da base64 o carica da path
        
        Args:
            image_data: Path file o dizionario con base64
            
        Returns:
            PIL Image o None se errore
        """
        try:
            if isinstance(image_data, str):
                # Path file
                if os.path.exists(image_data):
                    return Image.open(image_data).convert('RGB')
                else:
                    raise FileNotFoundError(f"File non trovato: {image_data}")
            
            elif isinstance(image_data, dict):
                # Base64 encoding
                if 'base64' in image_data:
                    img_data = base64.b64decode(image_data['base64'])
                    return Image.open(BytesIO(img_data)).convert('RGB')
                elif 'path' in image_data:
                    return Image.open(image_data['path']).convert('RGB')
            
            return None
            
        except Exception as e:
            print(f"⚠️  Errore decodifica immagine: {e}")
            return None
    
    def start_socket_server(self):
        """Avvia server socket per comunicazione diretta"""
        try:
            self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_server.bind(('localhost', self.socket_port))
            self.socket_server.listen(5)
            
            print(f"🔌 Socket server avviato su porta {self.socket_port}")
            
            while True:
                try:
                    client_socket, address = self.socket_server.accept()
                    print(f"📡 Connessione socket da {address}")
                    
                    # Gestisci richiesta in thread separato
                    thread = threading.Thread(
                        target=self._handle_socket_client,
                        args=(client_socket,)
                    )
                    thread.daemon = True
                    thread.start()
                    
                except Exception as e:
                    print(f"⚠️  Errore socket server: {e}")
                    
        except Exception as e:
            print(f"❌ Errore avvio socket server: {e}")
    
    def _handle_socket_client(self, client_socket):
        """Gestisci client socket"""
        try:
            # Ricevi dati
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b'\n\n' in data:  # Marker fine messaggio
                    break
            
            # Decodifica richiesta JSON
            request_data = json.loads(data.decode('utf-8').replace('\n\n', ''))
            
            # Processa richiesta
            response = self._process_images_request(request_data)
            
            # Invia risposta
            response_json = json.dumps(response) + '\n\n'
            client_socket.send(response_json.encode('utf-8'))
            
        except Exception as e:
            error_response = {
                'status': 'error',
                'error': str(e)
            }
            response_json = json.dumps(error_response) + '\n\n'
            client_socket.send(response_json.encode('utf-8'))
            
        finally:
            client_socket.close()
    
    def start_rest_server(self):
        """Avvia server REST Flask"""
        print(f"🌐 REST API server avviato su http://localhost:{self.port}")
        print("📚 Endpoints disponibili:")
        print(f"   GET  /health - Health check")
        print(f"   GET  /stats - Statistiche server")
        print(f"   GET  /system_info - Informazioni sistema")
        print(f"   POST /process_images - Elabora immagini")
        print(f"   POST /reset_stats - Reset statistiche")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
    
    def start_server(self, enable_socket: bool = True, enable_rest: bool = True):
        """
        Avvia il server con opzioni multiple
        
        Args:
            enable_socket: Abilita server socket
            enable_rest: Abilita server REST
        """
        print("🎯 Avvio server ViT Visual Servoing")
        
        if enable_socket:
            # Avvia socket server in thread separato
            self.socket_thread = threading.Thread(target=self.start_socket_server)
            self.socket_thread.daemon = True
            self.socket_thread.start()
        
        if enable_rest:
            # Avvia REST server (bloccante)
            self.start_rest_server()
        else:
            # Se solo socket, mantieni main thread attivo
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Server fermato dall'utente")
    
    def stop_server(self):
        """Ferma il server"""
        if self.socket_server:
            self.socket_server.close()
        print("🛑 Server fermato")


def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='ViT Visual Servoing Server')
    parser.add_argument('--config', type=str, default='vitvs_config.yaml',
                       help='Path al file di configurazione')
    parser.add_argument('--port', type=int, default=5000,
                       help='Porta REST API')
    parser.add_argument('--socket-port', type=int, default=6000,
                       help='Porta socket server')
    parser.add_argument('--no-rest', action='store_true',
                       help='Disabilita REST API')
    parser.add_argument('--no-socket', action='store_true',
                       help='Disabilita socket server')
    
    args = parser.parse_args()
    
    # Verifica configurazione
    config_path = args.config if os.path.exists(args.config) else None
    if config_path:
        print(f"📝 Usando configurazione: {config_path}")
    else:
        print("📝 Usando configurazione di default")
    
    try:
        # Inizializza server
        server = ViTVSServer(
            config_path=config_path,
            port=args.port,
            socket_port=args.socket_port
        )
        
        # Avvia server
        server.start_server(
            enable_socket=not args.no_socket,
            enable_rest=not args.no_rest
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server fermato dall'utente")
    except Exception as e:
        print(f"❌ Errore server: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
