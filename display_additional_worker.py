from loop_handler.advanced_loop_handler import AdditionalWorker
import numpy as np
from camera_parameter import CameraParameter
import cv2
from typing import Union

class DisplayAdditionalWorker(AdditionalWorker):

    def __init__(self,cam_param:CameraParameter):
        self.id = cam_param.id

    def run(self,foreground_frame:Union[np.ndarray,None],foreground_depth:Union[np.ndarray,None],background_frame:Union[np.ndarray,None],background_depth:Union[np.ndarray,None]):
        if foreground_frame is not None:
            if foreground_frame.shape[2] == 4:
                # アルファチャンネルを0〜1に正規化
                alpha = foreground_frame[:, :, 3] / 255.0

                result = np.zeros_like(foreground_frame[:, :, :3], dtype=np.uint8)
                for c in range(3):  # RGB各チャンネル
                    result[:, :, c] = foreground_frame[:, :, c] * alpha

                # 表示
                cv2.imshow(self.id, result)
                cv2.waitKey(1)