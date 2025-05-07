import random

import cv2
import numpy as np

from agent import Agent
from map import Map
from particle import Particle
from signals import ColorEnum
from utils import add, multiply, subtract


class Environment:
    def __init__(
            self,
            map: Map,
            agent: Agent,
            sensor_radius: int = 15
    ) -> None:
        self._map = map
        self._agent = agent
        self._swarm: list[Particle] = []
        self._sensor_radius: int = sensor_radius
        self._is_done: bool = False
        self._latest_percept: dict[str, np.ndarray|list[list[int]]] = {}
        self._latest_percept['view'] = np.full(self.map.img.shape, 100)
    
    @property
    def map(self) -> Map:
        return self._map
    
    def populate_swarm(self, count: int) -> None:
        starting_positions = self.map.find_positions(ColorEnum.BLUE)
        for particle in self._swarm:
            try:
                starting_positions.remove(particle.position)
            except ValueError:
                pass
        
        for _ in range(count):
            id_num = len(self._swarm)
            position = random.choice(starting_positions)
            starting_positions.remove(position)
            particle = Particle(
                id_num = id_num,
                position = position,
                velocity = [0 for _ in position],
                fitness_func = lambda _: 0
            )
            self._swarm.append(particle)
    
    def step(self) -> np.ndarray:
        percept = {}
        percept['view'] = self._latest_percept['view']
        percept['swarm_pos'] = [particle.position for particle in self._swarm]
        # percept['view'] = np.full(self.map.img.shape, 100)
        for particle in self._swarm:
            mask = [slice(max(axis - self._sensor_radius, 0),
                          axis + self._sensor_radius + 1)
                    for axis in particle.position]
            
            percept['view'][*mask] = self.map.img[*mask] # TODO: Fix conversion from [0,0,0] to ColorEnum
        
        self._latest_percept = percept
        fitness_func = self._agent.new_fitness_func(percept)
        self._swarm.sort(key=fitness_func, reverse=True)
        
        # TODO: set variabes
        inertia: float = random.random()
        exploration: float = random.random()
        exploitation: float = random.random()

        best_position = self._swarm[0].best_position
        for particle in self._swarm:
            new_velocity = add(
                multiply(inertia, particle.velocity),
                multiply(
                    exploration * random.random(),
                    subtract(
                        particle.best_position,
                        particle.position
                    )
                ),
                multiply(
                    exploitation * random.random(),
                    subtract(
                        best_position,
                        particle.position
                    )
                )
            )
            particle.move(new_velocity, fitness_func)
            percept['view'][*particle.position] = [000, 000, 255]
        
        return percept['view']
