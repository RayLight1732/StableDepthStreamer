from loop_handler.loop_handler import LoopHandler, LoopHandlerFactory
import cv2
import numpy as np
import time
from depth_predictor.depth_predictor import DepthPredictor
from stream_client.stream_client import StreamClient
from background_processor.background_processor import BackgroundProcessor
from foreground_processor.foreground_processor import ForegroundProcessor
from mask_caluculator.mask_calculator import MaskCalculator
from typing import Callable
from stream_client.png_data import PngData
from stream_client.raw_image_data import RawImageData
from stream_client.size_data import SizeData
import uuid
from util import (
    TYPE_BACKGROUND_DEPTH,
    TYPE_BACKGROUND_IMAGE,
    TYPE_FOREGROUND_DEPTH,
    TYPE_FOREGROUND_IMAGE,
)
from camera_parameter import CameraParameter


class AdvancedLoopHandler(LoopHandler):
    def __init__(
        self,
        id: str,
        predictor: DepthPredictor,
        client: StreamClient,
        fg_processor: ForegroundProcessor,
        bg_processor: BackgroundProcessor,
        mask_calculator: MaskCalculator,
    ):
        self.id = id
        self.predictor = predictor
        self.client = client
        self.fg_processor = fg_processor
        self.bg_processor = bg_processor
        self.mask_calculator = mask_calculator

    def on_loop(self, frame: cv2.typing.MatLike):
        start = time.time()
        self.client.send_data(SizeData(self.id, frame.shape[1], frame.shape[0]))

        depth = self.predictor.predict(frame)
        mask = self.mask_calculator.get_mask(
            frame, self.bg_processor.get_background()[0]
        )
        self.bg_processor.update_background(frame, depth, mask)

        if mask is not None:
            foreground_frame, foreground_depth = self.fg_processor.get_foreground(
                frame, depth, mask
            )
            background_frame, background_depth = self.bg_processor.get_background()
            foreground_id = str(uuid.uuid4())
            background_id = str(uuid.uuid4())
            if foreground_frame is not None:
                foreground_frame = cv2.cvtColor(foreground_frame, cv2.COLOR_BGRA2RGBA)
                foreground_frame_data = RawImageData(
                    self.id, TYPE_FOREGROUND_IMAGE, foreground_id, foreground_frame
                )
                self.client.send_data(foreground_frame_data)
            if foreground_depth is not None:
                foreground_depth_data = RawImageData(
                    self.id, TYPE_FOREGROUND_DEPTH, foreground_id, foreground_depth
                )
                self.client.send_data(foreground_depth_data)
            if background_frame is not None:
                background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
                background_frame_data = RawImageData(
                    self.id, TYPE_BACKGROUND_IMAGE, background_id, background_frame
                )
                self.client.send_data(background_frame_data)
            if background_depth is not None:
                background_depth_data = RawImageData(
                    self.id, TYPE_BACKGROUND_DEPTH, background_id, background_depth
                )
                self.client.send_data(background_depth_data)
        end = time.time()
        print(f"update takes {end-start}s")


class AdvancedLoopHandlerFactory(LoopHandlerFactory):
    def __init__(
        self,
        predictor: DepthPredictor,
        client: StreamClient,
        fg_processor_factory: Callable[[CameraParameter], ForegroundProcessor],
        bg_processor_factory: Callable[[CameraParameter], BackgroundProcessor],
        mask_calculator_factory: Callable[[CameraParameter], MaskCalculator],
    ):
        super().__init__()
        self.predictor = predictor
        self.client = client
        self.fg_processor_factory = fg_processor_factory
        self.bg_processor_factory = bg_processor_factory
        self.mask_calculator_factory = mask_calculator_factory

    def create(self, id: str) -> AdvancedLoopHandler:
        fg_processor = self.fg_processor_factory()
        bg_processor = self.bg_processor_factory()
        mask_calculator = self.mask_calculator_factory()
        return AdvancedLoopHandler(
            id, self.predictor, self.client, fg_processor, bg_processor, mask_calculator
        )
