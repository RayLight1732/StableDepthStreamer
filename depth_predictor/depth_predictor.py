from abc import ABCMeta, abstractmethod
import cv2
import numpy as np


class DepthPredictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, frame: cv2.typing.MatLike) -> np.ndarray:
        """
        予測を行う
        """
        pass
