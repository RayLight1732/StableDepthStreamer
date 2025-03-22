from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from typing import Union


class BackgroundProcessor(metaclass=ABCMeta):

    @abstractmethod
    def update_background(
        self,
        frame: cv2.typing.MatLike,
        depth: np.ndarray,
        mask: Union[np.ndarray, None],
    ):
        """
        背景を更新する
        initializedがfalseのとき、maskはNoneでもよい
        """
        pass

    @abstractmethod
    def get_background(self) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """背景画像及び深度を取得する"""
        pass

    @abstractmethod
    def initialized(self) -> bool:
        """初期化が完了したかどうかを返す"""
        pass
