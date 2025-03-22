import cv2
import time
import numpy as np
from typing import Union, Optional
from background_processor.background_processor import BackgroundProcessor


class NPModeBackgroundProcessor(BackgroundProcessor):
    def __init__(
        self, max_distance: float = 20, weight=1 / 1000, num_initial_frames=10
    ):
        """
        max_length:         最大距離
        weight:             新しいフレームの重み（未使用）
        num_initial_frames: 初期化に用いる画像の枚数
        """
        self.max_distance = max_distance
        self.background_frame = None
        self.background_depth = None
        self.frame_count = 0
        self.num_initial_frames = num_initial_frames
        self.weight = weight
        self.bg_histogram = None

    def update_background(
        self, frame: np.ndarray, depth: np.ndarray, mask: Union[np.ndarray, None]
    ):
        """
        マスクが1の部分を前景として、背景を加重平均で更新する
        """
        start = time.time()

        # 深度を 0~255 に正規化
        depth = np.clip(depth / self.max_distance * 255, 0, 255).astype(np.uint8)

        # ヒストグラムの初期化
        if self.bg_histogram is None:
            self.bg_histogram = np.zeros(depth.shape + (256,), dtype=np.float32)

        if self.frame_count < self.num_initial_frames:
            # 背景の初期化（フレーム平均）
            if self.background_frame is None:
                # 平均をとるため小数点も考慮してfloat32
                self.background_frame = frame.astype(np.float32)
                self.background_depth = depth.astype(np.float32)
            else:
                self.background_frame += frame
                self.background_depth += depth

            self.frame_count += 1

            if self.frame_count == self.num_initial_frames:
                # 初期背景を平均化
                # これ以降はuint8で大丈夫
                self.background_depth = (
                    self.background_depth / self.num_initial_frames
                ).astype(np.uint8)
                self.background_frame = (
                    self.background_frame / self.num_initial_frames
                ).astype(np.uint8)

                # ヒストグラムの初期値を設定
                x_indices, y_indices = np.indices(depth.shape)
                self.bg_histogram[x_indices, y_indices, self.background_depth] += 10
                self.max_bins = self.background_depth

            return

        if mask is None:
            return

        mask_gpu = np.asarray(mask)

        # mask == 0 のインデックスを取得
        update_indices = np.nonzero(mask_gpu == 0)
        # ヒストグラムの減衰
        self.bg_histogram[update_indices] *= 0.9

        depth_gup_update_indices = depth[update_indices]
        # 更新する場所は減衰させない
        self.bg_histogram[
            update_indices[0], update_indices[1], depth_gup_update_indices
        ] /= 0.9

        # ヒストグラムに新しいデータを追加
        self.bg_histogram[
            update_indices[0], update_indices[1], depth_gup_update_indices
        ] += 1
        # ヒストグラムで最頻値を取得
        self.max_bins = np.argmax(self.bg_histogram, axis=2).astype(np.uint8)

        # 背景フレームを更新
        match_pixels = depth == self.max_bins
        update_pixels = match_pixels & (mask_gpu == 0)
        self.background_frame[update_pixels] = frame[update_pixels]  # type: ignore

        end = time.time()
        print(f"mode filter takes {end - start}s")

    def get_background(self) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """背景画像及び深度を取得する"""
        if not self.initialized():
            return (None, None)
        return (
            self.background_frame.astype(np.uint8),  # type: ignore
            self.max_bins.astype(np.float32) / 255 * self.max_distance,
        )

    def initialized(self) -> bool:
        """初期化が完了したかどうかを返す"""
        return self.frame_count >= self.num_initial_frames
