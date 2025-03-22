from stream_client.serializable_data import SerializableData
import cv2
import numpy as np


class RawImageData(SerializableData):
    def __init__(self, camera_id: str, type: int, id: str, image: cv2.typing.MatLike):
        """
        image: RGB or RGBA or R
        """
        self.camera_id = camera_id.encode()
        self.camera_id_size = len(self.camera_id).to_bytes(4, byteorder="little")
        self.type = type.to_bytes(4, byteorder="little")
        self.id = id.encode()
        self.id_size = len(self.id).to_bytes(4, byteorder="little")
        self.width = image.shape[1].to_bytes(4, byteorder="little")
        self.height = image.shape[0].to_bytes(4, byteorder="little")
        # unityのtexture2Dは原点が左下(!) C#では遅そう(ホント?)だからこちらで行う
        self.image = np.flipud(image).tobytes()
        self.image_size = len(self.image).to_bytes(4, byteorder="little")

    def to_bytes(self) -> bytes:
        return (
            self.camera_id_size
            + self.camera_id
            + self.type
            + self.id_size
            + self.id
            + self.width
            + self.height
            + self.image_size
            + self.image
        )

    def name(self) -> str:
        return "RawImageData"
