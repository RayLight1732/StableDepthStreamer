import dataclasses
import yaml


@dataclasses.dataclass
class CameraParameter:
    fx: float
    fy: float
    cx: float
    cy: float
    id: str
    index: int


def load_camera_params(path: str) -> list[CameraParameter]:
    with open(path) as f:
        data = yaml.safe_load(f)
        camera_parameters: list[CameraParameter] = [
            CameraParameter(**cam) for cam in data["camera_parameters"]
        ]

        return camera_parameters
