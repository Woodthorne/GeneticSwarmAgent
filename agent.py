import heapq
import random
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from map import AbstractMap, ImgMap
from signals import ColorEnum
from utils import euclidean, get_color, collision


STEP_SIZE = 10


class Agent:
    def __init__(
            self,
            map_shape: tuple[int],
            target_area: np.ndarray,
            map_type: AbstractMap|ImgMap
    ) -> None:
        self._map_shape = map_shape
        self._target_area = target_area
        self._map_type = map_type
    
    @property
    def target_area(self) -> np.ndarray:
        return self._target_area.copy()

    def new_fitness_func(self, percept: dict) -> Callable:
        destination = random.choice(self._target_area)
        origin = np.sum(percept['swarm_pos'], axis = 0) / percept['swarm_pos'].shape[1]
        if self._map_type == ImgMap:
            goal = self.a_star(origin, destination, percept['view'])
        else:
            goal = self.a_star(origin, destination, percept['obstacles'])
        # TODO: Use genetics to determine fitness based on percept
        def func(position: Iterable) -> float:
            return -euclidean(position, goal)
        return func
    
    def a_star(self, origin: np.ndarray, destination: np.ndarray, map_: np.ndarray):
        frontier = []
        heapq.heappush(frontier, (0, origin))
        prev_pos = {tuple(origin): None}
        
        current_cost = defaultdict(lambda: float('inf'))
        current_cost[tuple(origin)] = 0

        reachable_dest = None
        min_distance = float('inf')

        while frontier:
            _, current = heapq.heappop(frontier)
            distance = euclidean(current, destination)
            if distance < min_distance:
                reachable_dest = current
                min_distance = distance
            
            if tuple(current) == tuple(destination):
                break

            for movement in [[0, -STEP_SIZE], [0, STEP_SIZE], [-STEP_SIZE, 0], [STEP_SIZE, 0]]:
                neighbor = current + movement
                if not all([0 for _ in self._map_shape] < neighbor) \
                or not all(neighbor < self._map_shape):
                    continue

                if self._map_type == ImgMap:
                    location = map_[*neighbor]
                    if get_color(location) == ColorEnum.BLACK:
                        continue
                else:
                    for obstacle in map_:
                        if collision((current, neighbor), obstacle):
                            continue
                    # intersect, position = collision()
                    # result = np.any(neighbor == map_, axis=0)
                    # print(result)

                new_cost = current_cost[tuple(current)] + 1
                if new_cost < current_cost[tuple(neighbor)]:
                    current_cost[tuple(neighbor)] = new_cost
                    priority = new_cost + euclidean(neighbor, destination)
                    heapq.heappush(frontier, (priority, neighbor))
                    prev_pos[tuple(neighbor)] = current
        
            route = []
            current = destination
            while np.all(current):
                route.append(current)
                current = prev_pos.get(tuple(current))
            route.reverse()

            if not route or np.all(route[0] != origin):
                route = [origin]
                if np.all(reachable_dest):
                    route.append(reachable_dest)
            
            return route