from stream_client.serializable_data import SerializableData
import cv2


class SizeData(SerializableData):
    def __init__(self, camera_id: str, width: int, height: int):
        self.camera_id = camera_id.encode()
        self.camera_id_size = len(self.camera_id).to_bytes(4, byteorder="little")
        self.width = width.to_bytes(4, byteorder="little")
        self.height = height.to_bytes(4, byteorder="little")

    def to_bytes(self) -> bytes:
        return self.camera_id_size + self.camera_id + self.width + self.height

    def name(self) -> str:
        return "SizeData"
