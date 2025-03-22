from foreground_processor.foreground_processor import ForegroundProcessor
import cv2
import numpy as np
from typing import Union


class Uint8ForegroundProcessor(ForegroundProcessor):
    def __init__(self, max_distance):
        self.max_distance = max_distance

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
        if mask is None:
            return None, None
        depth = (depth / self.max_distance * 255).astype(np.uint8)
        # open cvでは0~255のマスクを利用する
        mask = (mask * 255).astype(np.uint8)
        foreground_depth = cv2.bitwise_and(depth, depth, mask=mask)

        # アルファチャンネルを追加したBGRA画像を作成
        bgra = cv2.merge((frame, mask))

        return bgra, foreground_depth
