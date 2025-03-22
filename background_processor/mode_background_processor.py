import cv2
import time
import numpy as np
import cupy as cp
from typing import Union, Optional
from background_processor.background_processor import BackgroundProcessor


class ModeBackgroundProcessor(BackgroundProcessor):
    def __init__(
        self, max_distance: float = 20, weight=1 / 1000, num_initial_frames=10
    ):
        """
        max_distance:         最大距離
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

        frame_gpu = cp.asarray(frame)

        # 深度を 0~255 に正規化
        depth = np.clip(depth / self.max_distance * 255, 0, 255).astype(np.uint8)
        # GPUに転送
        depth_gpu = cp.asarray(depth, dtype=cp.uint8)

        # ヒストグラムの初期化
        if self.bg_histogram is None:
            self.bg_histogram = cp.zeros(depth_gpu.shape + (256,), dtype=cp.float32)

        if self.frame_count < self.num_initial_frames:
            # 背景の初期化（フレーム平均）
            if self.background_frame is None:
                # 平均をとるため小数点も考慮してfloat32
                self.background_frame = frame_gpu.astype(cp.float32)
                self.background_depth = depth_gpu.astype(cp.float32)
            else:
                self.background_frame += frame_gpu
                self.background_depth += depth_gpu

            self.frame_count += 1

            if self.frame_count == self.num_initial_frames:
                # 初期背景を平均化
                # これ以降はuint8で大丈夫
                self.background_depth = (
                    self.background_depth / self.num_initial_frames
                ).astype(cp.uint8)
                self.background_frame = (
                    self.background_frame / self.num_initial_frames
                ).astype(cp.uint8)

                # ヒストグラムの初期値を設定
                x_indices, y_indices = cp.indices(depth_gpu.shape)
                self.bg_histogram[x_indices, y_indices, self.background_depth] += 10
                self.max_bins = self.background_depth

            return

        if mask is None:
            return

        mask_gpu = cp.asarray(mask)

        # mask == 0 のインデックスを取得
        update_indices = cp.nonzero(mask_gpu == 0)
        # ヒストグラムの減衰
        self.bg_histogram[update_indices] *= 0.99

        depth_gup_update_indices = depth_gpu[update_indices]
        # 更新する場所は減衰させない
        self.bg_histogram[
            update_indices[0], update_indices[1], depth_gup_update_indices
        ] /= 0.99

        # ヒストグラムに新しいデータを追加
        self.bg_histogram[
            update_indices[0], update_indices[1], depth_gup_update_indices
        ] += 1
        # ヒストグラムで最頻値を取得
        self.max_bins = cp.argmax(self.bg_histogram, axis=2).astype(cp.uint8)

        # 背景フレームを更新
        match_pixels = depth_gpu == self.max_bins
        update_pixels = match_pixels & (mask_gpu == 0)
        self.background_frame[update_pixels] = frame_gpu[update_pixels]  # type: ignore

        end = time.time()
        print(f"mode filter takes {end - start}s")

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

    def get_background(self) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """背景画像及び深度を取得する"""
        if not self.initialized():
            return (None, None)
        return (
            cp.asnumpy(self.background_frame.astype(cp.uint8)),  # type: ignore
            cp.asnumpy(self.max_bins.astype(cp.uint8)),
        )

    def initialized(self) -> bool:
        """初期化が完了したかどうかを返す"""
        return self.frame_count >= self.num_initial_frames
