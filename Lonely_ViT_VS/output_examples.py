#!/usr/bin/env python3
"""
Script di esempio per mostrare l'output esatto restituito dal server.

Questo script dimostra:
1. L'output formato JSON completo
2. Come parsare e utilizzare i dati
3. Gestione degli errori
4. Esempi pratici per tutti i tipi di response
"""

import json
import time
from typing import Dict, Any

def simulate_success_response() -> Dict[str, Any]:
    """Simula una response di successo tipica del server."""
    return {
        "status": "success",
        "velocity": [0.1234, -0.0567, 0.0890, 0.0123, -0.0456, 0.0789],
        "num_features": 15,
        "velocity_norm": 0.1654,
        "method": "vit_standalone",
        "iteration": 1,
        "processing_time": 0.234,
        "request_id": 42,
        "goal_points": [[123.4, 567.8], [234.5, 678.9], [345.6, 789.0]],
        "current_points": [[125.6, 570.2], [236.7, 681.1], [347.8, 791.2]]
    }

def simulate_error_response() -> Dict[str, Any]:
    """Simula una response di errore tipica del server."""
    return {
        "status": "error",
        "error": "File non trovato: current_image.jpg",
        "processing_time": 0.012,
        "request_id": 43
    }

def simulate_health_response() -> Dict[str, Any]:
    """Simula una response di health check."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": "cuda",
        "uptime": 123.45
    }

def simulate_stats_response() -> Dict[str, Any]:
    """Simula una response di stats."""
    return {
        "requests_count": 156,
        "successful_requests": 142,
        "failed_requests": 14,
        "total_processing_time": 45.67,
        "average_processing_time": 0.293,
        "uptime": 3600.12,
        "server_start_time": 1640995200.123
    }

def parse_success_response(response: Dict[str, Any]) -> None:
    """Esempio di come parsare una response di successo."""
    print("🎯 SUCCESS RESPONSE PARSING:")
    print("=" * 50)
    
    # Check status
    if response['status'] != 'success':
        print("❌ Response non è success!")
        return
    
    # Estrai velocità IBVS
    velocity = response['velocity']
    print(f"📐 Velocity Vector: {velocity}")
    print(f"   Linear velocities:  vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
    print(f"   Angular velocities: wx={velocity[3]:.4f}, wy={velocity[4]:.4f}, wz={velocity[5]:.4f}")
    
    # Metriche di qualità
    print(f"🔍 Features detected: {response['num_features']}")
    print(f"📊 Velocity norm: {response['velocity_norm']:.4f}")
    print(f"⚡ Processing time: {response['processing_time']:.3f}s")
    print(f"🔢 Request ID: {response['request_id']}")
    
    # Punti (se presenti)
    if 'goal_points' in response and 'current_points' in response:
        goal_points = response['goal_points']
        current_points = response['current_points']
        print(f"📍 Feature Points ({len(goal_points)} pairs):")
        for i, (goal, current) in enumerate(zip(goal_points, current_points)):
            print(f"   Pair {i+1}: Goal({goal[0]:.1f}, {goal[1]:.1f}) → Current({current[0]:.1f}, {current[1]:.1f})")
    
    print()

def parse_error_response(response: Dict[str, Any]) -> None:
    """Esempio di come gestire una response di errore."""
    print("❌ ERROR RESPONSE PARSING:")
    print("=" * 50)
    
    if response['status'] != 'error':
        print("⚠️  Response non è error!")
        return
    
    print(f"💥 Error: {response['error']}")
    print(f"⚡ Processing time: {response['processing_time']:.3f}s")
    print(f"🔢 Request ID: {response['request_id']}")
    
    # Gestione errori specifica
    error_msg = response['error'].lower()
    if 'file non trovato' in error_msg:
        print("💡 Suggestion: Verifica che il file esista e sia leggibile")
    elif 'feature' in error_msg or 'corrispondenza' in error_msg:
        print("💡 Suggestion: Prova con immagini con più dettagli visivi")
    elif 'formato' in error_msg:
        print("💡 Suggestion: Usa formati supportati (jpg, png, bmp)")
    
    print()

def parse_health_response(response: Dict[str, Any]) -> None:
    """Esempio di come interpretare health check."""
    print("❤️  HEALTH RESPONSE PARSING:")
    print("=" * 50)
    
    status = response.get('status', 'unknown')
    model_loaded = response.get('model_loaded', False)
    device = response.get('device', 'unknown')
    uptime = response.get('uptime', 0)
    
    print(f"🟢 Server Status: {status}")
    print(f"🧠 Model Loaded: {'Yes' if model_loaded else 'No'}")
    print(f"🖥️  Device: {device.upper()}")
    print(f"⏱️  Uptime: {uptime:.1f}s ({uptime/60:.1f} min)")
    
    if status == 'healthy' and model_loaded:
        print("✅ Server è pronto per elaborazioni!")
    else:
        print("⚠️  Server potrebbe avere problemi")
    
    print()

def parse_stats_response(response: Dict[str, Any]) -> None:
    """Esempio di come interpretare statistiche server."""
    print("📊 STATS RESPONSE PARSING:")
    print("=" * 50)
    
    total_requests = response.get('requests_count', 0)
    success_requests = response.get('successful_requests', 0)
    failed_requests = response.get('failed_requests', 0)
    avg_time = response.get('average_processing_time', 0)
    uptime = response.get('uptime', 0)
    
    success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
    
    print(f"📈 Total Requests: {total_requests}")
    print(f"✅ Successful: {success_requests} ({success_rate:.1f}%)")
    print(f"❌ Failed: {failed_requests}")
    print(f"⚡ Average Processing Time: {avg_time:.3f}s")
    print(f"⏱️  Server Uptime: {uptime/3600:.1f} hours")
    
    # Performance insights
    if avg_time < 0.1:
        print("🚀 Performance: Excellent")
    elif avg_time < 0.3:
        print("👍 Performance: Good")
    else:
        print("⚠️  Performance: Could be improved")
    
    print()

def demonstrate_client_code() -> None:
    """Dimostra come usare l'output nel codice client."""
    print("💻 CLIENT CODE EXAMPLES:")
    print("=" * 50)
    
    # Simula response
    response = simulate_success_response()
    
    print("# Python Client Example:")
    print("if response['status'] == 'success':")
    print("    velocity = response['velocity']")
    print("    # Usa velocity per controllo robot")
    print("    robot.set_velocity(velocity)")
    print("    ")
    print("    # Monitor quality")
    print(f"    if response['num_features'] < 10:")
    print("        print('Warning: Low feature count')")
    print()
    
    print("# Real-time control loop example:")
    print("while not converged:")
    print("    response = server.process_images(goal, current)")
    print("    if response['status'] == 'success':")
    print("        velocity = response['velocity']")
    print("        velocity_norm = response['velocity_norm']")
    print("        ")
    print("        # Check convergence")
    print("        if velocity_norm < 0.01:")
    print("            converged = True")
    print("            break")
    print("        ")
    print("        # Apply control")
    print("        robot.move(velocity)")
    print("    else:")
    print("        print(f\"Error: {response['error']}\")")
    print("        break")
    print()

def main():
    """Funzione principale che dimostra tutti i tipi di output."""
    print("🤖 ViT Visual Servoing - Server Output Examples")
    print("=" * 60)
    print()
    
    # Success Response
    success_response = simulate_success_response()
    print("📄 RAW SUCCESS RESPONSE JSON:")
    print(json.dumps(success_response, indent=2))
    print()
    parse_success_response(success_response)
    
    # Error Response  
    error_response = simulate_error_response()
    print("📄 RAW ERROR RESPONSE JSON:")
    print(json.dumps(error_response, indent=2))
    print()
    parse_error_response(error_response)
    
    # Health Response
    health_response = simulate_health_response()
    print("📄 RAW HEALTH RESPONSE JSON:")
    print(json.dumps(health_response, indent=2))
    print()
    parse_health_response(health_response)
    
    # Stats Response
    stats_response = simulate_stats_response()
    print("📄 RAW STATS RESPONSE JSON:")
    print(json.dumps(stats_response, indent=2))
    print()
    parse_stats_response(stats_response)
    
    # Client Code Examples
    demonstrate_client_code()
    
    print("🎯 SUMMARY:")
    print("=" * 50)
    print("✅ Il server restituisce sempre JSON strutturato")
    print("✅ Field 'status' indica successo/errore")
    print("✅ Response include sempre processing_time e request_id")
    print("✅ Velocity è array [vx,vy,vz,wx,wy,wz] per controllo diretto")
    print("✅ Punti sono opzionali (include_points=true)")
    print("✅ Metriche di qualità (num_features, velocity_norm)")
    print("✅ Gestione errori dettagliata con messaggi specifici")
    print()
    print("📖 Per dettagli completi: OUTPUT_FORMAT.md")

if __name__ == '__main__':
    main()
