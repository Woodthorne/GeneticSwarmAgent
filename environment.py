import random

import cv2
import numpy as np

from agent import Agent
from map import AbstractMap, ImgMap
from drone import Drone
from signals import ColorEnum
from utils import intersection, obstacle_in_view, get_discrete_coords, merge_vectors, dev_print

class Environment:
    def __init__(
            self,
            map: AbstractMap|ImgMap,
            agent: Agent,
            sensor_radius: int = 10,
            collisions: bool = True
    ) -> None:
        self._map = map
        self._agent = agent
        self._sensor_radius = sensor_radius
        self._collisions = collisions
        
        self._swarm: list[Drone] = []
        self._is_done: bool = False
        self._latest_percept: dict[str, np.ndarray] = {}
        self._detections: list[np.ndarray] = []
        if isinstance(self._map, ImgMap):
            self._latest_percept['view'] = np.full(self._map.img.shape, 100)
        else:
            self._latest_percept['obstacles'] = []
    
    @property
    def is_done(self) -> bool:
        return self._is_done
    
    def populate_swarm(self, count: int) -> None:
        if isinstance(self._map, ImgMap):
            starting_positions = self._map.find_positions(ColorEnum.BLUE)
            for drone in self._swarm:
                mask = (starting_positions != drone.position).any(axis = 1)
                starting_positions = starting_positions[mask]
        
        for _ in range(count):
            id_num = len(self._swarm)
            if isinstance(self._map, ImgMap):
                position = random.choice(starting_positions)
            
                mask = (starting_positions != position).any(axis = 1)
                starting_positions = starting_positions[mask]
            else:
                occupied_positions = np.zeros((0, len(self._map.axes)))
                for drone in self._swarm:
                    occupied_positions = np.vstack([occupied_positions, drone.position])
                while True:
                    position = np.array(
                        [random.random() * (ax1 - ax0) + ax0
                         for ax0, ax1 in zip(*self._map.start)]
                    )
                    if np.all(position not in occupied_positions):
                        break
            
            drone = Drone(
                id_num = id_num,
                position = position,
                velocity = np.zeros([len(position)], dtype=np.int64),
                fitness_func = lambda _: 0
            )
            self._swarm.append(drone)
    
    def step(self) -> np.ndarray:
        self._swarm.sort(key=lambda p: p.fitness)
        dev_print('location', self._swarm[0].position)
        dev_print('swarm_fitness', [int(p.fitness) for p in self._swarm])
        percept = {}
        percept['swarm_pos'] = np.array([drone.position for drone in self._swarm])
        percept['swarm_data'] = np.array([(drone.position, drone.velocity, drone.best_position) for drone in self._swarm])
        
        if isinstance(self._map, ImgMap):
            percept['view'] = self._latest_percept['view'].copy()
            # percept['view'] = np.full(self._map.img.shape, 100)
            for drone in self._swarm:
                mask = [slice(max(axis - self._sensor_radius, 0),
                            axis + self._sensor_radius + 1)
                        for axis in drone.position]
                percept['view'][*mask] = self._map.img[*mask] # TODO: Fix conversion from [0,0,0] to ColorEnum
        else:
            detections: list[np.ndarray] = []
            for drone in self._swarm:
                for obstacle in self._map.obstacles:
                    count, coordinates = obstacle_in_view(obstacle, drone.position, self._sensor_radius)
                    if count == 2:
                        detections.append(coordinates)
            
            self._detections.extend(detections)
            self._detections = merge_vectors(self._detections)

            percept['obstacles'] = np.array(self._map.obstacles)
            # percept['obstacles'] = np.array(self._detections)
            percept['obstacles'] = np.unique(percept['obstacles'], axis = 0)
            print(f'Currently tracking {len(percept['obstacles'])} obstacles.')
        
        self._latest_percept = percept
        fitness_func, inertia, exploration, exploitation = self._agent.new_fitness_func(percept)
        
        best_position = self._swarm[0].best_position
        frame = np.full((*self._map.axes, 3), 255)
        for drone in self._swarm:
            new_velocity = (
                inertia * drone.velocity \
                + exploration * random.random() * (drone.best_position - drone.position) \
                + exploitation * random.random() * (best_position - drone.position)
            ).astype(np.int64)
            move_vector = drone.move(new_velocity, fitness_func)
            if self._collisions and isinstance(self._map, ImgMap):
                if not all([0 for _ in self._map.axes] < drone.position) \
                or not all(drone.position < self._map.axes):
                    print(f'Collision occured at {drone.position}')
                    cv2.waitKey(0) & 0xFF == ord('q')
                    quit()
            elif self._collisions:
                for obstacle in self._map.obstacles:
                    intersect, position = intersection(move_vector, obstacle)
                    if intersect:
                        print(f'Collision occured at {position}')
                        dev_print(move_vector)
                        dev_print(percept['obstacles'])
                        cv2.waitKey(0) & 0xFF == ord('q')
                        quit()
            if isinstance(self._map, ImgMap):
                try:
                    percept['view'][*drone.position] = [0, 0, 255]
                except IndexError:
                    pass
            else:

                frame[*drone.position.astype(dtype=np.int8)] = [0, 0, 255]

        if isinstance(self._map, ImgMap):
            if all(drone.position in self._agent.target_area for drone in self._swarm):
                self._is_done = True
            
            return percept['view']
        else:
            if all(self._map.in_goal(drone.position) for drone in self._swarm):
                self._is_done = True

            detections = []
            for obstacle in percept['obstacles']:
                points = get_discrete_coords(obstacle)
                detections.append((obstacle, points))
                for point in points:
                    try:
                        frame[*point] = [0, 0, 0]
                    except IndexError:
                        print('obstacle drawing issue')
                        print(points)
                        print(point)
                        quit()

                # frame[*obstacle.astype(dtype=np.int8)] = [0, 0, 0]
            # print(detections)
            # cv2.imshow('frame', frame.astype(np.uint8))
            # cv2.waitKey(0) & 0xFF == ord('q')
            # quit()

            return frame