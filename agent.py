import csv
import heapq
import random
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from utils import euclidean, intersection, to_tuple, dev_print, manhattan


INFINITY = float('inf')


class Percept:
    swarm_data: np.ndarray
    obstacles: np.ndarray


class Agent:
    def __init__(
            self,
            target_area: np.ndarray,
            step_size: float
    ) -> None:
        self._target_area = target_area
        self._checkpoint = None
        self._step_size = step_size
    
    @property
    def target_area(self) -> np.ndarray:
        return self._target_area.copy()
    
    def random_target(self) -> np.ndarray:
        x0, x1 = sorted(self.target_area[:, 0])
        y0, y1 = sorted(self.target_area[:, 1])
        x = x0 + random.random() * (x1 - x0)
        y = y0 + random.random() * (y1 - y0)
        return np.array((x, y))

    def new_fitness_func(self, percept: Percept) -> Callable:
        sum_of_points = np.sum(percept.swarm_data[:, 0, :] + percept.swarm_data[:, 1, :], axis = 0)
        num_of_points = percept.swarm_data.shape[0]
        swarm_center = sum_of_points / num_of_points
        destination = self.random_target()
        
        if self._checkpoint is None:
            checkpoint_distance = INFINITY
        else:
            checkpoint_distance = euclidean(self._checkpoint, destination)
        
        swarm_distance = euclidean(swarm_center, destination)
        if 0.9 * swarm_distance < checkpoint_distance:
            dev_print('Finding route')
            # route = self.a_star(swarm_center,
            route = self.a_star(swarm_center,
                                destination,
                                percept.obstacles)
            dev_print('Route found')
            if len(route) > 1:
                self._checkpoint = route[1]
            else:
                self._checkpoint = route[0]
            # self._checkpoint = destination
            # dev_print('new checkpoint:', self._checkpoint)
        
        def fitness_func(position: Iterable) -> float:
            distance = euclidean(position, self._checkpoint)
            return distance
        
        dev_print('Finding params')
        genetic_iee = self.genetic_params(
            percept.swarm_data,
            percept.obstacles,
            fitness_func
        )
        dev_print('Params found')

        return fitness_func, genetic_iee
    

    def a_star(self, origin: np.ndarray, destination: np.ndarray, obstacles: list[np.ndarray]) -> list[np.ndarray]:
        src_node = to_tuple(origin)
        dst_node = to_tuple(destination)
        open_nodes = []
        closed_nodes = []
        
        def heuristic(node: np.ndarray) -> float:
            # distance = manhattan(node, destination)
            distance = euclidean(node, destination)
            return distance

        previous_node = {src_node: None}
        travelled_distances = defaultdict(lambda :INFINITY)
        
        def f_value(node: tuple) -> float:
            g = travelled_distances[node]
            h = heuristic(node)
            return g + h

        travelled_distances[src_node] = 0

        heapq.heappush(open_nodes, (f_value(src_node), src_node))
        while open_nodes:
            _, current_node = heapq.heappop(open_nodes)
            if current_node == dst_node:
                break

            closed_nodes.append(current_node)
            current_array = np.array(current_node) 
            step_size = min(self._step_size,
                            euclidean(current_array, destination))
            x_mods = [-1, -1, -1, 0, 1, 1, 1, 0]
            y_mods = [-1, 0, 1, 1, 1, 0, -1, -1]
            xy_mods = np.array([(x, y) for x, y in zip(x_mods, y_mods)])
            movements = step_size * xy_mods
            for movement in movements:
                neighbor_array = current_array + movement
                neighbor_node = to_tuple(neighbor_array)
                if neighbor_node in closed_nodes:
                    continue

                blocked = False
                travel_vector = np.array([current_array, neighbor_array])
                for obstacle in obstacles:
                    blocked, _ = intersection(travel_vector, obstacle)
                    if blocked:
                        break
                if blocked:
                    continue

                new_distance = travelled_distances[current_node] + step_size
                if new_distance < travelled_distances[neighbor_node]:
                    travelled_distances[neighbor_node] = new_distance
                    heapq.heappush(open_nodes, (f_value(neighbor_node), neighbor_node))
                    previous_node[neighbor_node] = current_node
        
        route = []
        if current_node == dst_node:
            while current_node:
                route.append(np.array(current_node))
                current_node = previous_node.get(current_node)
            route.reverse()

        # with open('route.csv', 'w', encoding='utf-8') as file:
        #     writer = csv.writer(file, lineterminator='\r')
        #     writer.writerow(['x', 'y'])
        #     for step in route:
        #         writer.writerow([*step])
        # quit()
        # dev_print(route)
        return route
  
    def genetic_params(
            self,
            swarm_data: np.ndarray,
            obstacles: np.ndarray,
            global_fitness_func: Callable
    ) -> tuple[float, float, float]:
        rows, *_, dims = swarm_data.shape
        best_position = np.full((rows, dims), swarm_data[:, 2, :][0])
        best_position = self._checkpoint

        def genetic_fitness(genome: tuple[float, float, float]) -> float:
            data = swarm_data.copy()
            new_positions = data[:, 0, :] + data[:, 1, :]
            
            iee = np.array(genome).reshape((rows, 3))
            inertia, exploration, exploitation = np.hsplit(iee, 3)
            
            data[:, 1, :] = (
                inertia * data[:, 1, :] \
                + exploration * (data[:, 2, :] - data[:, 0, :]) \
                + exploitation * (best_position - data[:, 0, :])
            )
            
            data[:, 0, :] = new_positions
            data[:, 1, :] = data[:, 0, :] + data[:, 1, :]
            swarm_center = np.sum(data[:, 0, :], axis = 0) / data.shape[-1]
            fitness = global_fitness_func(swarm_center)
            fitness = 0
            for vector in data[:, 0:2, :]:
                overstep = euclidean(*vector) > self._step_size
                if overstep:
                    return INFINITY
                
                collisions = [intersection(vector, obstacle)[0]
                              for obstacle in obstacles]
                if any(collisions):
                    # dev_print('BAD GENE', genome)
                    return INFINITY
                
                fitness += global_fitness_func(vector[1])
                # fitness += euclidean(vector[1], swarm_center)
            
            # dev_print('GOOD GENE', genome)
            
            return fitness
    
        generation = 0
        population_size = 50
        population = [[random.random() * random.random()
                       for _ in range(3 * rows)]
                      for _ in range(population_size)]
        
        best_fitness = genetic_fitness(population[0])
        repeated_scores = 0
        repetition_limit = 10
        stability = 0.5
        resets = 0

        while repeated_scores < repetition_limit or best_fitness == INFINITY:
            generation += 1
            
            population.sort(key=genetic_fitness)
            if genetic_fitness(population[0]) == INFINITY:
                resets += 1
                dev_print(f'Genomes reset: {resets}')
                factor = 1
                for _ in range(resets):
                    factor *= random.random()
                population = [[random.random() * factor
                               for _ in range(3 * rows)]
                               for _ in range(population_size)]
                continue
            
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
                        new_genome.append(random.random() * random.random())
                new_population.append(tuple(new_genome))
            
            population = new_population
            new_fitness = genetic_fitness(population[0])
            if best_fitness > new_fitness:
                best_fitness = new_fitness
                repeated_scores = 0
            else:
                repeated_scores += 1
        
        iee = np.array(population[0]).reshape(rows, 3)
        
        # # TEMP TODO: Figure out why collision happens
        # data = swarm_data.copy()
        # new_positions = data[:, 0, :] + data[:, 1, :]
            
        # inertia, exploration, exploitation = np.hsplit(iee, 3)
        # # inertia, exploration, exploitation = genome
        
        # data[:, 1, :] = (
        #     inertia * data[:, 1, :] \
        #     + exploration * (data[:, 2, :] - data[:, 0, :]) \
        #     + exploitation * (best_position - data[:, 0, :])
        # )
        
        # data[:, 0, :] = new_positions
        # data[:, 1, :] = data[:, 0, :] + data[:, 1, :]
        # for vector in data[:, 0:2, :]:
        #     collisions = [intersection(vector, obstacle)[0]
        #                     for obstacle in obstacles]
        #     dev_print(f'{vector.flatten()=}')
            
        return iee

    # def a_star_LEGACY(
    #         self,
    #         origin: np.ndarray,
    #         destination: np.ndarray,
    #         obstacles: np.ndarray
    # ) -> list[np.ndarray]:
    #     t_origin = to_tuple(origin)
    #     t_destination = to_tuple(destination)
    #     frontier = []
    #     heapq.heappush(frontier, (0, t_origin))
    #     prev_pos = {t_origin: None}
        
    #     current_cost = defaultdict(lambda: INFINITY)
    #     current_cost[t_origin] = 0

    #     t_reachable_dest = None
    #     min_distance = INFINITY
    #     step_size = max(self._step_size, manhattan(origin, destination) // 10)

    #     while frontier:
    #         _, t_current = heapq.heappop(frontier)
    #         distance = manhattan(t_current, destination)
    #         if distance < min_distance:
    #             t_reachable_dest = t_current
    #             min_distance = distance
            
    #         if t_current == t_destination:
    #             break

    #         current = np.array(t_current)
    #         step_size = random.random() * self._step_size
    #         for movement in [[0, -step_size], [0, step_size],
    #                          [-step_size, 0], [step_size, 0]]:
    #             neighbor = current + movement
    #             # if not all([0 for _ in self._map_shape] < neighbor) \
    #             # or not all(neighbor < self._map_shape):
    #             #     continue

    #             blocked = False
    #             travel_vector = np.array([current, neighbor])
    #             for obstacle in obstacles:
    #                 if intersection(travel_vector, obstacle):
    #                     blocked = True
    #                     break
    #             if blocked:
    #                 continue

    #             new_cost = current_cost[t_current] + step_size
    #             if new_cost < current_cost[tuple(neighbor)]:
    #                 t_neighbor = to_tuple(neighbor)
    #                 current_cost[t_neighbor] = new_cost
    #                 priority = new_cost + manhattan(neighbor, destination)
    #                 heapq.heappush(frontier, (priority, t_neighbor))
    #                 prev_pos[t_neighbor] = current
        
    #     route = []
    #     t_current = t_destination
    #     while t_current:
    #         route.append(np.array(t_current))
    #         t_current = prev_pos.get(t_current)
    #     route.reverse()

    #     if not route or np.all(route[0] != origin):
    #         route = [origin]
    #         if t_reachable_dest:
    #             route.append(np.array(t_reachable_dest))
        
    #     return route
