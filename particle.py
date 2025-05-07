from typing import Callable

from utils import add

class Particle:
    def __init__(
            self,
            id_num: int,
            position: list[int],
            velocity: list[int],
            fitness_func: Callable
    ) -> None:
        self._id = id_num
        self._position = tuple(position)
        self._fitness_func = fitness_func
        self._fitness = fitness_func(position)
        self._best_position = tuple(position)
        self._best_fitness = self.fitness
        self._velocity = tuple(velocity)

    @property
    def id(self) -> int:
        return self._id

    @property
    def position(self) -> list[int]:
        return self._position
    
    @property
    def fitness(self) -> float:
        return self._fitness
    
    @property
    def best_position(self) -> list[int]:
        return self._best_position

    @property
    def best_fitness(self) -> list[int]:
        return self._best_fitness
    
    @property
    def velocity(self) -> list[int]:
        return self._velocity
    
    def move(self, new_velocity: list[float],new_fitness_func: Callable) -> None:
        self._position = add(self.position, new_velocity)
        self._velocity = new_velocity
        self._fitness_func = new_fitness_func
        self._fitness = new_fitness_func(self.position)
        if self.fitness > self.best_fitness:
            self._best_fitness = self.fitness
            self._best_position = self.position
        