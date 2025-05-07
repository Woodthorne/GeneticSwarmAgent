import random
from typing import Callable, Iterable

import numpy as np

from utils import distance


class Agent:
    def __init__(
            self,
            map_shape: tuple[int],
            target_area: np.ndarray
    ) -> None:
        self._map_shape = map_shape
        self._target_area = target_area
    
    @property
    def target_area(self) -> np.ndarray:
        return self._target_area.copy()

    def new_fitness_func(self, percept: dict) -> Callable:
        goal = random.choice(self._target_area)
        # TODO: Use genetics to determine fitness based on percept
        def func(position: Iterable) -> float:
            return -distance(position, goal)
        return func
    