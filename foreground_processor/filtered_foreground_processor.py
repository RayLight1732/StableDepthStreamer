from foreground_processor.foreground_processor import ForegroundProcessor
import cv2
import numpy as np
from typing import Union
from camera_parameter import CameraParameter
from sklearn.cluster import DBSCAN


class FilteredForegroundProcessor(ForegroundProcessor):

    def __init__(self, camera_param: CameraParameter, eps=0.1, min_samples=10):
        self.camera_param = camera_param
        self.eps = eps
        self.min_samples = min_samples

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

        fx, fy, cx, cy = (
            self.camera_param.fx,
            self.camera_param.fy,
            self.camera_param.cx,
            self.camera_param.cy,
        )

        y_indices, x_indices = mask.nonzero()
        z = depth[y_indices, x_indices]
        x = (x_indices - cx) * z / fx
        y = (y_indices - cy) * z / fy
        points = np.column_stack((x, y, z))
        if len(points) != 0:
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = clustering.labels_
            cluster_points = points[labels != -1]

            new_depth = np.zeros(shape=depth.shape, dtype=np.float32)

            x_meter: np.ndarray
            y_meter: np.ndarray
            z_meter: np.ndarray
            x_meter, y_meter, z_meter = cluster_points.T
            x_indices = ((x_meter / z_meter * fx) + cx).astype(int)
            y_indices = ((y_meter / z_meter * fy) + cy).astype(int)
            new_depth[y_indices, x_indices] = z_meter

            depth = new_depth

        alpha = (mask * 255).astype(np.uint8)
        bgra = cv2.merge((frame, alpha))
        return bgra, depth
