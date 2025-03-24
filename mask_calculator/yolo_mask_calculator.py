from mask_calculator.mask_calculator import MaskCalculator
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from PIL import Image
from torch import uint8
from typing import Union


class YOLOMaskCalculator(MaskCalculator):
    """
    YOLOを用いてマスクを計算する
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_mask(
        self,
        frame: cv2.typing.MatLike,
        background_frame: Union[cv2.typing.MatLike, None],
    ) -> Union[np.ndarray, None]:
        """前景を1,背景を0としたマスクを計算する"""
        # 推論画像を保存しない、0.25以上の精度のものを出力、logを出さない
        result = self.model.predict(
            frame, save=False, conf=0.25, verbose=False, retina_masks=True
        )[0]
        # 複数枚の推論に対応しているため、返り値は配列だが、今回は一枚のためインデックス0を使用
        is_obb = result.obb is not None
        boxes = result.obb if is_obb else result.boxes

        if boxes is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            box: Boxes
            for i, box in enumerate(boxes):  # type: ignore
                cls_id = box.cls
                cls_name = result.names[cls_id.item()]
                if cls_name == "person":
                    # numpyに変換
                    masks = result.masks
                    if masks is not None:
                        np_mask = masks.data[i].clone().to(uint8).cpu().detach().numpy()
                        # orを取る
                        mask |= np_mask

            return mask
        return None
