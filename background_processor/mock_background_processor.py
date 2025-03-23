from background_processor.background_processor import BackgroundProcessor
import cv2
import numpy as np
from typing import Union


class MockBackgroundProcessor(BackgroundProcessor):

    def __init__(self, max_distance=20):
        self.max_distance = 20
        self.frame = None
        self.depth = None

    def update_background(
        self,
        frame: cv2.typing.MatLike,
        depth: np.ndarray,
        mask: Union[np.ndarray, None],
    ):
        self.frame = frame
        self.depth = (depth / self.max_distance * 255).astype(np.uint8)

    def get_background(self) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """背景画像及び深度を取得する"""
        return self.frame, self.depth

    def initialized(self) -> bool:
        """初期化が完了したかどうかを返す"""
        return self.frame is not None and self.depth is not None
