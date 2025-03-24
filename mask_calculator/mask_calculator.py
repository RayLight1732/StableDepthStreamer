from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from typing import Union


class MaskCalculator(metaclass=ABCMeta):

    @abstractmethod
    def get_mask(
        self,
        frame: cv2.typing.MatLike,
        background_frame: Union[cv2.typing.MatLike, None],
    ) -> Union[np.ndarray, None]:
        """前景を1,背景を0としたマスクを計算する"""
        pass
