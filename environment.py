import random

import numpy as np

from agent import Agent
from map import Map
from particle import Particle
from signals import ColorEnum

class Environment:
    def __init__(
            self,
            map: Map,
            agent: Agent,
            sensor_radius: int = 5
    ) -> None:
        self._map = map
        self._agent = agent
        self._swarm: list[Particle] = []
        self._sensor_radius: int = sensor_radius
        self._is_done: bool = False
    
    @property
    def map(self) -> Map:
        return self._map
    
    def populate_swarm(self, count: int) -> None:
        starting_positions = self.map.find_positions(ColorEnum.BLUE)
        
        for particle in self._swarm:
            starting_positions.remove(particle.position)
        
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
        percept['view'] = np.full(self.map.img.shape[:-1], None)
        for particle in self._swarm:
            mask = [slice(max(axis - self._sensor_radius, 0), axis + self._sensor_radius + 1)
                    for axis in particle.position]
            
            percept['view'][*mask] = self.map.img[*mask] # TODO: Fix conversion from [0,0,0] to ColorEnum
            # print('position', particle.position)
            # print('mask', *mask)
            # print('ndim', self.map.img.ndim)
            # print(self.map.img[mask[0]])
            # print([val for val in range(-self._sensor_radius, self._sensor_radius + 1)])
            # quit()
            # mask = self.map.img[[axis-self._sensor_radius:axis+self._sensor_radius for axis in particle.position]]
        

        percept['swarm_pos'] = [particle.position for particle in self._swarm]
        print(percept)