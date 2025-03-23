from foreground_processor.foreground_processor import ForegroundProcessor
import cv2
import numpy as np
from typing import Union


class ScaledFloat32ForegroundProcessor(ForegroundProcessor):
    """
    depthをfloat32に変換する
    """

    def __init__(self, max_distance: float, processor: ForegroundProcessor):
        self.max_distance = max_distance
        self.processor = processor

    def get_foreground(
        self,
        frame: cv2.typing.MatLike,
        depth: np.ndarray,
        mask: Union[np.ndarray, None],
    ) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        マスクを適用した画像及び深度を計算する
        frane: 画像
        depth: 深度(単位:m)
        mask: 0,1の配列
        返り値: bgraの画像,深度は255をmax_distanceとしたuint8で返却される
        """
        bgra, foreground_depth = self.processor.get_foreground(frame, depth, mask)
        if foreground_depth is None:
            return bgra, foreground_depth

        foreground_depth = (foreground_depth / self.max_distance).astype(np.float32)

        return bgra, foreground_depth
