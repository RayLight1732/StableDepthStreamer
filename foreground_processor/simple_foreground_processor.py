from foreground_processor.foreground_processor import ForegroundProcessor
import cv2
import numpy as np
from typing import Union, Optional


class SimpleForegroundProcessor(ForegroundProcessor):

    def get_foreground(
        self,
        frame: cv2.typing.MatLike,
        depth: np.ndarray,
        mask: Union[np.ndarray, None],
    ) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        マスクを適用した画像及び深度を計算する
        返り値: bgraの画像,
        """
        if mask is None:
            return None, None
        
        # 以降では0~255のマスクを利用する
        mask = mask * 255
        foreground_depth = cv2.bitwise_and(depth, depth, mask=mask)

        # アルファチャンネルを追加したBGRA画像を作成
        alpha = mask.astype(np.uint8)
        bgra = cv2.merge((frame, alpha))

        return bgra, foreground_depth
