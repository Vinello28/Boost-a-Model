import socket
import cv2
import pickle
import struct
from dotenv import load_dotenv
import os

load_dotenv()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((os.getenv("CAMERA_FROM_IP"), os.getenv("CAMERA_PORT")))
server_socket.listen(5)
conn, addr = server_socket.accept()

data = b""
payload_size = struct.calcsize("Q")

while True:
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data)
    cv2.imshow("Remote Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
