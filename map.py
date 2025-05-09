from pathlib import Path

import cv2
import numpy as np

from signals import ColorEnum
from utils import Point, Vector, drill, get_color, inside_zone, sightline


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
        
    
class AbstractMap:
    def __init__(
            self,
            axes: tuple[int, ...],
            start_zone: Vector,
            goal_zone: Vector,
            obstacles: list[Vector]
    ):
        self._axes = axes
        self._start = np.array(start_zone)
        self._goal = np.array(goal_zone)
        self._obstacles = [np.array(vector) for vector in obstacles]
        border = [
                sightline((-1, -1), axes[0] + 1, 0),
                sightline((-1, -1), axes[1] + 1, 90),
                sightline(axes, axes[0] + 1, 180),
                sightline(axes, axes[1] + 1, 270)
        ]
        self._obstacles.extend(border)
        
    
    @property
    def axes(self) -> tuple[int, ...]:
        return self._axes
    
    @property
    def start(self) -> Vector:
        return self._start

    @property
    def goal(self) -> Vector:
        return self._goal
    
    def in_start(self, point: Point) -> bool:
        return inside_zone(point, self.start)

    def in_goal(self, point: Point) -> bool:
        return inside_zone(point, self.goal)

    