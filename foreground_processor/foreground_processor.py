from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from typing import Union


class ForegroundProcessor(metaclass=ABCMeta):

    @abstractmethod
    def get_foreground(
        self,
        frame: cv2.typing.MatLike,
        depth: np.ndarray,
        mask: Union[np.ndarray, None],
    ) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        マスク(前景が1,背景が0)を適用した画像及び深度を計算する
        返り値: bgraの画像,
        """
        pass
