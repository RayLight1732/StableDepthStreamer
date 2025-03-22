from abc import ABCMeta, abstractmethod
import cv2


class LoopHandler(metaclass=ABCMeta):
    @abstractmethod
    def on_loop(self, frame: cv2.typing.MatLike):
        """
        ループの処理を行う
        """
        pass


class LoopHandlerFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, id: str) -> LoopHandler:
        """
        新しいLoopHandlerを作成する
        id: 対象のカメラID
        """
        pass
