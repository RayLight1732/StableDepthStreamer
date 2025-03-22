from stream_client.serializable_data import SerializableData
import cv2


class PngData(SerializableData):
    def __init__(self, camera_id: str, type: int, id: str, image: cv2.typing.MatLike):
        self.camera_id = camera_id.encode()
        self.camera_id_size = len(self.camera_id).to_bytes(4, byteorder="little")
        self.type = type.to_bytes(4, byteorder="little")
        self.id = id.encode()
        self.id_size = len(self.id).to_bytes(4, byteorder="little")
        _, png_encoded = cv2.imencode(".png", image)
        self.image = png_encoded.tobytes()
        self.image_size = len(self.image).to_bytes(4, byteorder="little")

    def to_bytes(self) -> bytes:
        return (
            self.camera_id_size
            + self.camera_id
            + self.type
            + self.id_size
            + self.id
            + self.image_size
            + self.image
        )

    def name(self) -> str:
        return "PngData"
