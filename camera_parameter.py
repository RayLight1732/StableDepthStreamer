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
    width:int
    height:int


# camera_parameters:
#   - fx: 1000.0
#     fy: 1000.0
#     cx: 640.0
#     cy: 480.0
#     id: "camera_1"
#     index: 0

#   - fx: 950.0
#     fy: 950.0
#     cx: 620.0
#     cy: 460.0
#     id: "camera_2"
#     index: 1


def load_camera_params(path: str) -> list[CameraParameter]:
    with open(path) as f:
        data = yaml.safe_load(f)
        camera_parameters: list[CameraParameter] = [
            CameraParameter(**cam) for cam in data["camera_parameters"]
        ]

        return camera_parameters
