import random

import cv2
import numpy as np

from agent import Agent
from map import AbstractMap, ImgMap
from particle import Particle
from signals import ColorEnum
from utils import intersection, sightline

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
        
        self._swarm: list[Particle] = []
        self._is_done: bool = False
        self._latest_percept: dict[str, np.ndarray|list[list[int]]] = {}
        if isinstance(self._map, ImgMap):
            self._latest_percept['view'] = np.full(self._map.img.shape, 100)
        else:
            self._latest_percept['obstacles'] = np.zeros((0, len(self._map.axes)))
    
    @property
    def is_done(self) -> bool:
        return self._is_done
    
    def populate_swarm(self, count: int) -> None:
        if isinstance(self._map, ImgMap):
            starting_positions = self._map.find_positions(ColorEnum.BLUE)
            for particle in self._swarm:
                mask = (starting_positions != particle.position).any(axis = 1)
                starting_positions = starting_positions[mask]
        
        
        for _ in range(count):
            id_num = len(self._swarm)
            if isinstance(self._map, ImgMap):
                position = random.choice(starting_positions)
            
                mask = (starting_positions != position).any(axis = 1)
                starting_positions = starting_positions[mask]
            else:
                occupied_positions = np.zeros((0, len(self._map.axes)))
                for particle in self._swarm:
                    occupied_positions = np.vstack([occupied_positions, particle.position])
                while True:
                    position = np.array(
                        [random.random() * (ax1 - ax0) + ax0
                         for ax0, ax1 in zip(*self._map.start)]
                    )
                    if np.all(position not in occupied_positions):
                        break
            
            particle = Particle(
                id_num = id_num,
                position = position,
                velocity = np.zeros([len(position)], dtype=np.int64),
                fitness_func = lambda _: 0
            )
            self._swarm.append(particle)
    
    def step(self) -> np.ndarray:
        percept = {}
        percept['swarm_pos'] = np.array([particle.position for particle in self._swarm])
        if isinstance(self._map, ImgMap):
            percept['view'] = self._latest_percept['view'].copy()
            # percept['view'] = np.full(self._map.img.shape, 100)
            for particle in self._swarm:
                mask = [slice(max(axis - self._sensor_radius, 0),
                            axis + self._sensor_radius + 1)
                        for axis in particle.position]
                percept['view'][*mask] = self._map.img[*mask] # TODO: Fix conversion from [0,0,0] to ColorEnum
        else:
            percept['obstacles'] = self._latest_percept['obstacles'].copy()
            for particle in self._swarm:
                for angle in range(360):
                    sight_vector = sightline(particle.position, self._sensor_radius, angle)
                    for obstacle in self._map._obstacles:
                        intersect, point = intersection(sight_vector, obstacle)
                        if intersect:
                            percept['obstacles'] = np.vstack((percept['obstacles'], point))
            percept['obstacles'] = np.unique(percept['obstacles'], axis = 0)
        
        self._latest_percept = percept
        fitness_func = self._agent.new_fitness_func(percept)
        self._swarm.sort(key=lambda p: p.fitness, reverse=True)
        
        # TODO: set variabes
        inertia: float = random.random()
        exploration: float = random.random()
        exploitation: float = random.random()

        best_position = self._swarm[0].best_position
        # print(self._swarm[0].fitness)
        frame = np.full((*self._map.axes, 3), 255)
        for particle in self._swarm:
            new_velocity = (
                inertia * particle.velocity \
                + exploration * random.random() * (particle.best_position - particle.position) \
                + exploitation * random.random() * (best_position - particle.position)
            ).astype(np.int64)
            move_vector = particle.move(new_velocity, fitness_func)
            if self._collisions and isinstance(self._map, ImgMap):
                if not all([0 for _ in self._map.axes] < particle.position) \
                or not all(particle.position < self._map.axes):
                    print(f'Collision occured at {particle.position}')
                    cv2.waitKey(0) & 0xFF == ord('q')
                    quit()
            elif self._collisions:
                for obstacle in self._map._obstacles:
                    intersect, position = intersection(move_vector, obstacle)
                    if intersect:
                        print(f'Collision occured at {position}')
                        cv2.waitKey(0) & 0xFF == ord('q')
                        quit()
            if isinstance(self._map, ImgMap):
                try:
                    percept['view'][*particle.position] = [0, 0, 255]
                except IndexError:
                    pass
            else:

                frame[*particle.position.astype(dtype=np.int8)] = [0, 0, 255]

        if isinstance(self._map, ImgMap):
            if all(particle.position in self._agent.target_area for particle in self._swarm):
                self._is_done = True
            
            return percept['view']
        else:
            if all(self._map.in_goal(particle.position) for particle in self._swarm):
                self._is_done = True

            for obstacle in percept['obstacles']:
                frame[*obstacle.astype(dtype=np.int8)] = [0, 0, 0]
            return frame