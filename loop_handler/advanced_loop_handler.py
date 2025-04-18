from loop_handler.loop_handler import LoopHandler, LoopHandlerFactory
import cv2
import numpy as np
import time
from depth_predictor.depth_predictor import DepthPredictor
from stream_client.stream_client import StreamClient
from background_processor.background_processor import BackgroundProcessor
from foreground_processor.foreground_processor import ForegroundProcessor
from mask_calculator.mask_calculator import MaskCalculator
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
import logger
from camera_parameter import CameraParameter
from abc import ABCMeta, abstractmethod
from typing import Union


class AdditionalWorker(metaclass=ABCMeta):
    @abstractmethod
    def run(self,foreground_frame:Union[np.ndarray,None],foreground_depth:Union[np.ndarray,None],background_frame:Union[np.ndarray,None],background_depth:Union[np.ndarray,None]):
        pass

class AdvancedLoopHandler(LoopHandler):
    def __init__(
        self,
        id: str,
        predictor: DepthPredictor,
        client: StreamClient,
        fg_processor: ForegroundProcessor,
        bg_processor: BackgroundProcessor,
        mask_calculator: MaskCalculator,
        additional_worker:Union[AdditionalWorker,None] = None
    ):
        self.id = id
        self.predictor = predictor
        self.client = client
        self.fg_processor = fg_processor
        self.bg_processor = bg_processor
        self.mask_calculator = mask_calculator
        self.additional_worker = additional_worker

    def on_loop(self, frame: cv2.typing.MatLike):
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
        
            if self.additional_worker is not None:
                self.additional_worker.run(foreground_frame,foreground_depth,background_frame,background_depth)



class AdvancedLoopHandlerFactory(LoopHandlerFactory):
    def __init__(
        self,
        predictor_factory:Callable[[CameraParameter],DepthPredictor],
        client: StreamClient,
        fg_processor_factory: Callable[[CameraParameter], ForegroundProcessor],
        bg_processor_factory: Callable[[CameraParameter], BackgroundProcessor],
        mask_calculator_factory: Callable[[CameraParameter], MaskCalculator],
        additional_worker_factory:Union[Callable[[CameraParameter],AdditionalWorker],None] = None
    ):
        super().__init__()
        self.predictor_factory = predictor_factory
        self.client = client
        self.fg_processor_factory = fg_processor_factory
        self.bg_processor_factory = bg_processor_factory
        self.mask_calculator_factory = mask_calculator_factory
        self.additional_worker_factory = additional_worker_factory

    def create(self, cam_param:CameraParameter) -> AdvancedLoopHandler:
        predictor = self.predictor_factory(cam_param)
        fg_processor = self.fg_processor_factory(cam_param)
        bg_processor = self.bg_processor_factory(cam_param)
        mask_calculator = self.mask_calculator_factory(cam_param)
        additional_worker = self.additional_worker_factory(cam_param) if self.additional_worker_factory is not None else None
        return AdvancedLoopHandler(
            cam_param.id, predictor, self.client, fg_processor, bg_processor, mask_calculator,additional_worker
        )
    

