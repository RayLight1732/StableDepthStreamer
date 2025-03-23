import dataclasses


@dataclasses.dataclass
class CameraParameter:
    fx: float
    fy: float
    cx: float
    cy: float
