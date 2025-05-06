from pathlib import Path

import cv2
import numpy as np

from signals import ColorEnum
from utils import drill, get_color

class Map:
    def __init__(self, map_name: str) -> None:
        self._img_path = Path('maps') / map_name
        self._img: np.ndarray = cv2.imread(self._img_path)
        self._axes: int = len(self.img.shape) - 1

    @property
    def img(self) -> np.ndarray:
        return self._img.copy()
    
    @property
    def img_path(self) -> Path:
        return self._img_path

    @property
    def axes(self) -> int:
        return self._axes
    
    def check_position(self, position: list[int]) -> ColorEnum:
        found = self._img
        for axis in position:
            found = found[axis]

        return get_color(found)
    
    def find_positions(self, color: ColorEnum) -> list[list[int]]:
        positions = [position for position, pixel in drill(self._img)
                     if get_color(pixel) == color]
        return positions
    