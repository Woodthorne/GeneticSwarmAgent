import heapq
import random
from collections import defaultdict
from functools import cache
from typing import Callable, Iterable

import numpy as np

from map import AbstractMap, ImgMap
from signals import ColorEnum
from utils import euclidean, get_color, intersection, to_tuple


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
        self._checkpoint = None
        self._step_size = min(map_shape) // 10
    
    @property
    def target_area(self) -> np.ndarray:
        return self._target_area.copy()

    def new_fitness_func(self, percept: dict[str, np.ndarray]) -> Callable:
        swarm_center = np.sum(percept['swarm_data'][:, 0, :], axis = 0) / percept['swarm_data'].shape[-1]
        destination = random.choice(self._target_area)
        
        # print(percept['obstacles'])

        if np.any(self._checkpoint):
            check_point_distance = euclidean(self._checkpoint, destination)
        else:
            check_point_distance = float('inf')
        
        swarm_distance = euclidean(swarm_center, destination)
        if swarm_distance < check_point_distance:
            if self._map_type == ImgMap:
                route = self.a_star(swarm_center, destination, percept['view'])
            else:
                route = self.a_star(swarm_center, destination, percept['obstacles'])
            self._checkpoint = route[-1]
            print('new checkpoint:', self._checkpoint)
        
        def func(position: Iterable) -> float:
            return euclidean(position, self._checkpoint)
        
        inertia, exploration, exploitation = self.genetic_params(percept['swarm_data'], percept['obstacles'], func)
        
        return func, inertia, exploration, exploitation
    
    def a_star(self, origin: np.ndarray, destination: np.ndarray, map_: np.ndarray):
        t_origin = to_tuple(origin)
        t_destination = to_tuple(destination)
        frontier = []
        heapq.heappush(frontier, (0, t_origin))
        prev_pos = {t_origin: None}
        
        current_cost = defaultdict(lambda: float('inf'))
        current_cost[t_origin] = 0

        t_reachable_dest = None
        min_distance = float('inf')

        while frontier:
            # TODO: Figure out random crash during heappop
            # print('frontier', frontier[0])
            _, t_current = heapq.heappop(frontier)
            distance = euclidean(t_current, destination)
            if distance < min_distance:
                t_reachable_dest = t_current
                min_distance = distance
            
            if t_current == t_destination:
                break

            current = np.array(t_current)
            step_size = random.randint(0, self._step_size)
            for movement in [[0, -step_size], [0, step_size],
                             [-step_size, 0], [step_size, 0]]:
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
                        if intersection(np.array([current, neighbor]), obstacle):
                            continue

                new_cost = current_cost[t_current] + 1
                if new_cost < current_cost[tuple(neighbor)]:
                    t_neighbor = to_tuple(neighbor)
                    current_cost[t_neighbor] = new_cost
                    priority = new_cost + euclidean(neighbor, destination)
                    heapq.heappush(frontier, (priority, t_neighbor))
                    prev_pos[t_neighbor] = current
        
        route = []
        t_current = t_destination
        while t_current:
        # while np.all(t_current):
            route.append(np.array(t_current))
            t_current = prev_pos.get(t_current)
        route.reverse()

        if not route or np.all(route[0] != origin):
            route = [origin]
            if t_reachable_dest:
                route.append(np.array(t_reachable_dest))
        
        return route
        
    def genetic_params(self, drone_data: np.ndarray, obstacles: np.ndarray, global_fitness_func: Callable) -> tuple[float, float, float]:
        row, *_, dims = drone_data.shape
        best_position = np.full((row, dims), drone_data[0, 2, :][0])
        
        @cache
        def genetic_fitness(genome: tuple[float, float, float]) -> float:    
            inertia, exploration, exploitation = genome
            data = drone_data.copy()
            # print(data[0])
            data[:, 0, :] = data[:, 0, :] + data[:, 1, :]
            # print(data[0])

            data[:, 1, :] = (
                inertia * data[:, 1, :] \
                + exploration * (data[:, 2, :] - data[:, 0, :]) \
                + exploitation * (best_position - data[:, 0, :])
            ).astype(np.int64)
            # print(data[0])

            for vector in data[:, 0:2, :]:
                collisions = [intersection(vector, obstacle)
                              for obstacle in obstacles]
                if any(collisions):
                    return float('inf')
            
            swarm_center = np.sum(data[:, 0, :], axis = 0) / data.shape[-1]
            return global_fitness_func(swarm_center)
    
        generation = 0
        population_size = 20
        population = [tuple(random.random() for _ in range(3))
                      for _ in range(population_size)]

        best_fitness = genetic_fitness(population[0])
        repeated_scores = 0
        repetition_limit = 10
        stability = 0.9

        while repeated_scores < repetition_limit:
            generation += 1
            population.sort(key=genetic_fitness)
            
            legacy_size = len(population) // 10
            pairing_size = len(population) // 2

            new_population = population[:legacy_size]
            while len(new_population) < len(population):
                pair = random.sample(population[:pairing_size], 2)
                new_genome = []
                for val_a, val_b in zip(*pair):
                    prob = random.random()
                    if prob < stability / 2:
                        new_genome.append(val_a)
                    elif prob < stability:
                        new_genome.append(val_b)
                    else:
                        new_genome.append(random.random())
                new_population.append(tuple(new_genome))
            
            population = new_population
            new_fitness = genetic_fitness(population[0])
            if best_fitness > new_fitness:
                best_fitness = new_fitness
                repeated_scores = 0
            else:
                repeated_scores += 1
        
        return population[0]
