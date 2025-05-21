import random

import cv2
import numpy as np

from agent import Agent, Percept
from map import AbstractMap
from drone import Drone
from utils import intersection, segment_circle_intersection, get_discrete_coords, merge_segments, dev_print

class Environment:
    def __init__(
            self,
            map: AbstractMap,
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
        self._latest_percept = Percept()
        self._detections: list[np.ndarray] = []
    
    @property
    def is_done(self) -> bool:
        return self._is_done
    
    def populate_swarm(self, count: int) -> None:
        for _ in range(count):
            id_num = len(self._swarm)
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
        percept = Percept()
        percept.swarm_data = np.array(
            [(drone.position, drone.velocity, drone.best_position)
             for drone in self._swarm]
        )

        detections: list[np.ndarray] = []
        for drone in self._swarm:
            for obstacle in self._map.obstacles:
                intersects, segment = segment_circle_intersection(
                    obstacle, drone.position, self._sensor_radius
                )
                if intersects == 2:
                    detections.append(segment)
        
        self._detections.extend(detections)
        self._detections = merge_segments(self._detections)

        # percept.obstacles = np.array(self._map.obstacles)
        percept.obstacles = np.array(self._detections)
        percept.obstacles = np.unique(percept.obstacles, axis = 0)
        print(f'Currently tracking {len(percept.obstacles)} obstacles.')
        
        self._latest_percept = percept
        fitness_func, genetic_iees = self._agent.new_fitness_func(percept)

        best_position = self._agent._checkpoint
        frame = np.full((*self._map.axes, 3), 255)
        for drone, iee in zip(self._swarm, genetic_iees):
            inertia, exploration, exploitation = iee
            new_velocity = (
                inertia * drone.velocity \
                + exploration * (drone.best_position - drone.position) \
                + exploitation * (best_position - drone.position)
            )
            move_vector = drone.move(new_velocity, fitness_func)
            
            for obstacle in self._map.obstacles:
                intersect, position = intersection(move_vector, obstacle)
                if intersect:
                    print(f'Collision occured at {position}')
                    dev_print('move', move_vector.flatten())
                    dev_print('obstacle:', obstacle.flatten())
                    trajectory = get_discrete_coords(move_vector)
                    for point in trajectory:
                        if np.all(point < self._map.axes):
                            frame[*point] = [0, 0, 255]
                    cv2.imshow('collision', frame.astype(np.uint8))
                    cv2.waitKey(0) & 0xFF == ord('q')
                    quit()
            
            frame[*drone.position.astype(dtype=np.int8)] = [0, 0, 255]

        if all(self._map.in_goal(drone.position) for drone in self._swarm):
            self._is_done = True

        detections = []
        for obstacle in percept.obstacles:
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
        
        # TEMP
        sum_of_points = np.sum([drone.position for drone in self._swarm], axis = 0)
        num_of_points = len(self._swarm)
        swarm_center: np.ndarray = sum_of_points / num_of_points
        # frame[*swarm_center.astype(np.int8)] = [120, 120, 120]
        frame[*self._agent._checkpoint.astype(np.int8)] = [255, 0, 0]
        # /TEMP

        return frame