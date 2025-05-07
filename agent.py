from typing import Callable, Iterable


class Agent:
    def __init__(self):
        pass

    def new_fitness_func(self, percept: dict) -> Callable:
        # TODO: Use genetics to determine fitness based on percept
        def func(position: Iterable) -> float:
            fitness = 0

            return fitness
        return func