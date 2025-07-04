from pathlib import Path

import cv2
import numpy as np

from signals import ColorEnum
from utils import Vector, Segment, drill, get_color, inside_zone


class AbstractMap:
    def __init__(
            self,
            axes: tuple[int, ...],
            start_zone: Segment,
            goal_zone: Segment,
            obstacles: list[Segment]
    ):
        self._axes = axes
        self._start = np.array(start_zone)
        self._goal = np.array(goal_zone)
        self._obstacles = [np.array(vector) for vector in obstacles]
        rows, cols = axes
        border = [
                np.array([(0, 0), (0, cols - 1)]),
                np.array([(0, 0), (rows - 1, 0)]),
                np.array([(rows - 1, cols - 1), (rows - 1, 0)]),
                np.array([(rows - 1, cols - 1), (0, cols - 1)])
        ]
        self._obstacles.extend(border)
        self._obstacles.reverse()
    
    @property
    def axes(self) -> tuple[int, ...]:
        return self._axes
    
    @property
    def start(self) -> np.ndarray:
        return self._start

    @property
    def goal(self) -> np.ndarray:
        return self._goal
    
    @property
    def obstacles(self) -> list[np.ndarray]:
        return self._obstacles.copy()
    
    def in_start(self, point: Vector) -> bool:
        return inside_zone(point, self.start)

    def in_goal(self, point: Vector) -> bool:
        return inside_zone(point, self.goal)


############################ UNUSED ################################


class ImgMap:
    def __init__(self, map_name: str) -> None:
        self._img_path = Path('maps') / map_name
        self._img: np.ndarray = cv2.imread(self._img_path)
        self._axes: tuple[int, ...] = self.img.shape[:-1]

    @property
    def img(self) -> np.ndarray:
        return self._img.copy()
    
    @property
    def img_path(self) -> Path:
        return self._img_path

    @property
    def axes(self) -> tuple[int, ...]:
        return self._axes
    
    def check_position(self, position: list[int]) -> ColorEnum:
        return get_color(self._img[*position])
    
    def find_positions(self, color: ColorEnum) -> np.ndarray:
        positions = [position for position, pixel in drill(self._img)
                     if get_color(pixel) == color]
        
        return np.array(positions)
