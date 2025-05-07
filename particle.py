from typing import Callable

import numpy as np

class Particle:
    def __init__(
            self,
            id_num: int,
            position: np.ndarray,
            velocity: np.ndarray,
            fitness_func: Callable
    ) -> None:
        self._id = id_num
        self._position = position
        self._fitness_func = fitness_func
        self._fitness = fitness_func(position)
        self._best_position = position
        self._best_fitness = self.fitness
        self._velocity = velocity

    @property
    def id(self) -> int:
        return self._id

    @property
    def position(self) -> np.ndarray:
        return self._position
    
    @property
    def fitness(self) -> float:
        return self._fitness
    
    @property
    def best_position(self) -> np.ndarray:
        return self._best_position

    @property
    def best_fitness(self) -> float:
        return self._best_fitness
    
    @property
    def velocity(self) -> np.ndarray:
        return self._velocity
    
    def move(self, new_velocity: list[float],new_fitness_func: Callable) -> None:
        self._position = self.position + self._velocity
        self._velocity = new_velocity
        self._fitness_func = new_fitness_func
        self._fitness = new_fitness_func(self.position)
        if self.fitness > self.best_fitness:
            self._best_fitness = self.fitness
            self._best_position = self.position
        