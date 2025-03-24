from depth_predictor.depth_predictor import DepthPredictor
import cv2
import numpy as np
import torch
import depth_pro
import logger
import uuid
from camera_parameter import CameraParameter

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("use gpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("use mps")
    else:
        print("use upc")
    return device


class MLDepthProDepthPredictor(DepthPredictor):
    """
    ML-Depth-Proを用いた実装
    """

    def __init__(self,camera_param:CameraParameter):

        model, transform = depth_pro.create_model_and_transforms(
            device=get_torch_device(),
            precision=torch.half,
        )
        model.eval()
        self.model = model
        self.transform = transform
        self.id = camera_param.id

    def predict(self, frame: cv2.typing.MatLike) -> np.ndarray:
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transform(imageRGB)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        torch.cuda.synchronize()
        logger.logger.start_process(self.id,"predict")
        prediction = self.model.infer(image)
        torch.cuda.synchronize()
        logger.logger.end_process(self.id,"predict")
        # logger.logger.start_process(self.id,"detach")
        # result = prediction["depth"].detach()
        # logger.logger.end_process(self.id,"detach")
        # logger.logger.start_process(self.id,"cpu")
        # result = result.cpu()
        # logger.logger.end_process(self.id,"cpu")
        # logger.logger.start_process(self.id,"numpy")
        # result = result.numpy()
        # logger.logger.end_process(self.id,"numpy")
        # logger.logger.start_process(self.id,"squeeze")
        # result = result.squeeze()
        # logger.logger.end_process(self.id,"squeeze")
        torch.cuda.synchronize()
        logger.logger.start_process(self.id,"to cpu")
        result = prediction["depth"].detach().cpu().numpy().squeeze()
        torch.cuda.synchronize()
        logger.logger.end_process(self.id,"to cpu")
        return result
