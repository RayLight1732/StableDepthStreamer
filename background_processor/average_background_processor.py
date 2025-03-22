import cv2
import numpy as np
from background_processor.background_processor import BackgroundProcessor
from typing import Union


class AverageBackgroundProcessor(BackgroundProcessor):
    def __init__(self, weight=1 / 1000, num_initial_frames=10):
        """
        weight:             新しいフレームの重み
        num_initial_frames: 初期化に用いる画像の枚数
        """
        self.background_frame = None
        self.background_depth = None
        self.frame_count = 0
        self.num_initial_frames = num_initial_frames
        self.weight = weight

    def update_background(
        self,
        frame: cv2.typing.MatLike,
        depth: np.ndarray,
        mask: Union[np.ndarray, None],
    ):
        """
        マスクが1の部分を前景として、背景を加重平均で更新する
        initializedがfalseのとき、maskはNoneでもよい
        """
        if self.frame_count < self.num_initial_frames:
            # 背景初期化フェーズ
            if self.background_frame is None:
                self.background_frame = frame.astype(np.float32)
                self.background_depth = depth.astype(np.float32)
            else:
                self.background_frame += frame.astype(np.float32)
                self.background_depth += depth.astype(np.float32)  # type: ignore

            self.frame_count += 1

            if self.frame_count == self.num_initial_frames:
                self.background_frame /= self.num_initial_frames
                self.background_depth /= self.num_initial_frames
        elif mask is not None:
            # 加重平均 `(9*previous_bg + new_bg) / 10`
            self.background_frame = np.where(
                mask[..., None] == 0,  # maskが0の部分を対象
                (1 - self.weight) * self.background_frame  # type: ignore
                + self.weight * frame.astype(np.float32),  # 加重平均
                self.background_frame,  # maskが1の部分は更新しない # type: ignore
            )

            self.background_depth = np.where(
                mask == 0,
                (9 * self.background_depth + depth.astype(np.float32)) / 10,  # type: ignore
                self.background_depth,  # type: ignore
            )

    def get_background(self) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """背景画像及び深度を取得する"""
        if not self.initialized():
            return (None, None)
        return self.background_frame.astype(np.uint8), self.background_depth.astype(  # type: ignore
            np.float32
        )

    def initialized(self) -> bool:
        """初期化が完了したかどうかを返す"""
        return self.frame_count >= self.num_initial_frames
