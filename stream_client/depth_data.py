from stream_client.serializable_data import SerializableData
import numpy as np


class DepthData(SerializableData):
    def __init__(
        self, camera_id: str, type: int, id: str, max_distance: int, depth: np.ndarray
    ):
        self.camera_id = camera_id.encode()
        self.camera_id_size = len(self.camera_id).to_bytes(4, byteorder="little")
        self.type = type.to_bytes(4, byteorder="little")
        self.id = id.encode()
        self.id_size = len(self.id).to_bytes(4, byteorder="little")
        self.max_distance = max_distance.to_bytes(4, byteorder="little")
        self.depth = depth.astype(np.float32).tobytes()
        self.depth_size = len(self.depth).to_bytes(4, byteorder="little")

    def to_bytes(self) -> bytes:
        return (
            self.camera_id_size
            + self.camera_id
            + self.type
            + self.id_size
            + self.id
            + self.depth_size
            + self.depth
        )

    def name(self) -> str:
        return "DepthData"
