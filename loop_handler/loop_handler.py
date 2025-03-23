from abc import ABCMeta, abstractmethod
import cv2
from camera_parameter import CameraParameter


class LoopHandler(metaclass=ABCMeta):
    @abstractmethod
    def on_loop(self, frame: cv2.typing.MatLike):
        """
        ループの処理を行う
        """
        pass


class LoopHandlerFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, parameter: CameraParameter) -> LoopHandler:
        """
        新しいLoopHandlerを作成する
        parameter: 対象のカメラパラメータ
        """
        pass
