#!/usr/bin/env python3
import cv2
import socket
import argparse
import logging
import numpy as np
from threading import Event

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraReceiver:
    def __init__(self, host, port, window_name="Camera Stream"):
        self.host = host
        self.port = port
        self.window_name = window_name
        self.stop_event = Event()
        
    def start(self):
        """Start receiving and displaying frames"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        logger.info(f"Waiting for connection on {self.host}:{self.port}")
        
        conn, addr = sock.accept()
        logger.info(f"Connected to {addr}")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        try:
            while not self.stop_event.is_set():
                # Receive frame size (first 4 bytes)
                size_data = conn.recv(4)
                if not size_data:
                    break
                    
                frame_size = int.from_bytes(size_data, 'big')
                
                # Receive frame data
                frame_data = bytearray()
                while len(frame_data) < frame_size:
                    packet = conn.recv(frame_size - len(frame_data))
                    if not packet:
                        break
                    frame_data.extend(packet)
                
                if len(frame_data) != frame_size:
                    logger.warning("Incomplete frame received")
                    continue
                
                # Convert to numpy array and decode
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow(self.window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
        finally:
            cv2.destroyAllWindows()
            conn.close()
            sock.close()
            logger.info("Connection closed")
    
    def stop(self):
        """Stop the receiver"""
        self.stop_event.set()

def main():
    parser = argparse.ArgumentParser(description='Camera Stream Receiver')
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=9000, help='Port to listen on')
    args = parser.parse_args()
    
    receiver = CameraReceiver(args.host, args.port)
    
    try:
        receiver.start()
    except KeyboardInterrupt:
        logger.info("Stopping by user request")
        receiver.stop()

if __name__ == '__main__':
    main()