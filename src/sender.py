#!/usr/bin/env python3
import cv2
import numpy as np
import socket
import argparse
import time
import logging
from threading import Event


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraSender:
    def __init__(self, camera_device, resolution, framerate, host, port, ssh_tunnel=False):
        self.camera_device = camera_device
        self.resolution = resolution
        self.framerate = framerate
        self.host = host
        self.port = port
        self.ssh_tunnel = ssh_tunnel
        self.stop_event = Event()
        
    def connect(self):
        """Establish connection to receiver"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def capture_and_send(self):
        """Capture frames and send to receiver"""
        cap = cv2.VideoCapture(self.camera_device)
        if not cap.isOpened():
            logger.error(f"Cannot open camera device {self.camera_device}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.framerate)
        
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Encode frame as JPEG
                _, jpeg_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = jpeg_frame.tobytes()
                
                # Send frame size (4 bytes) and frame data
                try:
                    self.sock.sendall(len(frame_data).to_bytes(4, 'big'))
                    self.sock.sendall(frame_data)
                except Exception as e:
                    logger.error(f"Send failed: {e}")
                    break
                
                # Control framerate
                time.sleep(1/self.framerate)
                
        finally:
            cap.release()
            self.sock.close()
            logger.info("Camera and connection closed")
    
    def stop(self):
        """Stop the streaming"""
        self.stop_event.set()

def main():
    parser = argparse.ArgumentParser(description='V4L2 Camera Sender')
    parser.add_argument('--camera', default='/dev/video0', help='V4L2 camera device')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--host', default='localhost', help='Receiver host')
    parser.add_argument('--port', type=int, default=9000, help='Port to connect to')
    args = parser.parse_args()
    
    sender = CameraSender(
        camera_device=args.camera,
        resolution=(args.width, args.height),
        framerate=args.fps,
        host=args.host,
        port=args.port
    )
    
    if sender.connect():
        try:
            sender.capture_and_send()
        except KeyboardInterrupt:
            logger.info("Stopping by user request")
            sender.stop()
    else:
        logger.error("Failed to establish connection")

if __name__ == '__main__':
    main()