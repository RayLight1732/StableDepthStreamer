from mask_caluculator.mask_calculator import MaskCalculator
import cv2
import numpy as np
from typing import Union


class DiffMaskCalculator(MaskCalculator):
    def __init__(self, threshold):
        """
        threshold:閾値(0~1)
        """
        self.threshold = threshold

    def get_mask(
        self,
        frame: cv2.typing.MatLike,
        background_frame: Union[cv2.typing.MatLike, None],
    ) -> Union[np.ndarray, None]:
        """背景との差分から前景を1,背景を0としたマスクを計算する"""
        if background_frame is None:
            return None  # 背景が未確定ならNoneを返す

        # 差分を計算
        diff = cv2.absdiff(frame, background_frame.astype(np.uint8))

        # グレースケール化 & 二値化
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, self.threshold * 255, 1, cv2.THRESH_BINARY)

        return mask
