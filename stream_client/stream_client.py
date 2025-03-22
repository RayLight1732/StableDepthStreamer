import socket
import threading
import time
import cv2
import numpy as np
import sys
from stream_client.serializable_data import SerializableData


class DataSender:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.lock = threading.Lock()

    def send(self, data: SerializableData):
        try:
            with self.lock:
                encoded = data.name().encode()
                size = len(encoded)
                size_header = size.to_bytes(4, byteorder="little")
                self.sock.sendall(size_header)
                self.sock.sendall(encoded)
                self.sock.sendall(data.to_bytes())
        except Exception as e:
            print(f"Failed to send data: {e}")


class StreamClient:
    def __init__(self, host="127.0.0.1", port=12345):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.sender = None

    def connect(self) -> bool:
        with self.lock:
            if self.connected:
                return False

            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.connected = True
                self.running = True
                self.sender = DataSender(self.sock)
                print("Connected to server")

                self.thread = threading.Thread(
                    target=self._monitor_connection, daemon=True
                )
                self.thread.start()
                return True
            except Exception as e:
                print(f"Connection failed: {e}")
                return False

    def disconnect(self):
        with self.lock:
            self.running = False
            if self.sock:
                self.sock.close()
                self.sock = None
            self.connected = False
            self.sender = None
            print("Disconnected from server")

        if self.thread:
            self.thread.join()

    def _monitor_connection(self):
        """サーバーとの接続を監視し、切断されたら再接続を試みる"""
        while self.running:
            try:
                if self.sock:
                    data = self.sock.recv(1024)
                    if not data:
                        print("Connection lost, attempting to reconnect...")
                        self.connected = False
                        self.sock.close()
                        self.sock = None
                        self.sender = None
                        self._reconnect()
            except (socket.error, OSError) as e:
                print(f"Error detected: {e}, reconnecting...")
                self.connected = False
                self.sock = None
                self._reconnect()
            time.sleep(1)

    def _reconnect(self):
        """一定間隔で再接続を試みる"""
        while self.running and not self.connected:
            print("Reconnecting...")
            time.sleep(5)
            self.connect()

    def send_data(self, data: SerializableData):
        if self.sender:
            self.sender.send(data)
