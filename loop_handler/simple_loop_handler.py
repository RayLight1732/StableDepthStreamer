from loop_handler.loop_handler import LoopHandler, LoopHandlerFactory
import cv2
import numpy as np
from depth_predictor.depth_predictor import DepthPredictor
from stream_client.stream_client import StreamClient
import time
import uuid
from stream_client.size_data import SizeData
from stream_client.raw_image_data import RawImageData
from util import TYPE_BACKGROUND_DEPTH, TYPE_BACKGROUND_IMAGE


class SimpleLoopHandler(LoopHandler):
    def __init__(self, id: str, predictor: DepthPredictor, client: StreamClient):
        self.predictor = predictor
        self.client = client
        self.id = id

    def on_loop(self, frame: cv2.typing.MatLike):
        start = time.time()
        self.client.send_data(SizeData(self.id, frame.shape[1], frame.shape[0]))

        depth = self.predictor.predict(frame)
        depth = (depth / 20 * 255).astype(np.uint8).astype(np.float32) / 255 * 20
        id = str(uuid.uuid4())
        frame_data = RawImageData(self.id, TYPE_BACKGROUND_IMAGE, id, frame)
        self.client.send_data(frame_data)
        depth_data = RawImageData(self.id, TYPE_BACKGROUND_DEPTH, id, depth)
        self.client.send_data(depth_data)
        end = time.time()
        print(f"update takes {end-start}s")


class SimpleLoopHandlerFactory(LoopHandlerFactory):
    def __init__(self, predictor: DepthPredictor, client: StreamClient):
        self.client = client
        self.predictor = predictor

    def create(self, id: str) -> SimpleLoopHandler:
        return SimpleLoopHandler(id, self.predictor, self.client)
