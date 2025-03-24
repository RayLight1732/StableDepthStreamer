import cv2
import numpy as np
from typing import Union
from mask_calculator.mask_calculator import MaskCalculator

class MockMaskCaluculator(MaskCalculator):

    def __init__(self, is_all_foreground: bool):
        self.is_all_foreground = is_all_foreground

    def get_mask(
        self,
        frame: cv2.typing.MatLike,
        background_frame: Union[cv2.typing.MatLike, None],
    ) -> Union[np.ndarray, None]:
        """前景を1,背景を0としたマスクを計算する"""
        if self.is_all_foreground:
            return np.ones(frame.shape[:2],np.uint8)
        else:
            return np.zeros(frame.shape[:2],np.uint8)
