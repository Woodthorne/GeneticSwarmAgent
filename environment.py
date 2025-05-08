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
        self._latest_percept['view'] = np.full(self._map.img.shape, 100)
    
    @property
    def is_done(self) -> bool:
        return self._is_done
    
    def populate_swarm(self, count: int) -> None:
        starting_positions = self._map.find_positions(ColorEnum.BLUE)
        for particle in self._swarm:
            mask = (starting_positions != particle.position).any(axis = 1)
            starting_positions = starting_positions[mask]
        
        for _ in range(count):
            id_num = len(self._swarm)
            position = random.choice(starting_positions)
            
            mask = (starting_positions != position).any(axis = 1)
            starting_positions = starting_positions[mask]
            
            particle = Particle(
                id_num = id_num,
                position = position,
                velocity = np.zeros([len(position)], dtype=np.int64),
                fitness_func = lambda _: 0
            )
            self._swarm.append(particle)
    
    def step(self) -> np.ndarray:
        percept = {}
        percept['view'] = self._latest_percept['view']
        
        percept['swarm_pos'] = np.array([particle.position for particle in self._swarm])
        # percept['view'] = np.full(self._map.img.shape, 100)
        for particle in self._swarm:
            mask = [slice(max(axis - self._sensor_radius, 0),
                          axis + self._sensor_radius + 1)
                    for axis in particle.position]
            percept['view'][*mask] = self._map.img[*mask] # TODO: Fix conversion from [0,0,0] to ColorEnum
        
        self._latest_percept = percept
        fitness_func = self._agent.new_fitness_func(percept)
        self._swarm.sort(key=lambda p: p.fitness, reverse=True)
        
        # TODO: set variabes
        inertia: float = random.random()
        exploration: float = random.random()
        exploitation: float = random.random()

        best_position = self._swarm[0].best_position
        print(self._swarm[0].fitness)
        for particle in self._swarm:
            new_velocity = (
                inertia * particle.velocity \
                + exploration * random.random() * (particle.best_position - particle.position) \
                + exploitation * random.random() * (best_position - particle.position)
            ).astype(np.int64)
            particle.move(new_velocity, fitness_func)
            if self._collisions:
                assert all([0 for _ in self._map.axes] < particle.position) \
                     and all(particle.position < self._map.axes), \
                        f'Collision at {particle.position}'
            try:
                percept['view'][*particle.position] = [0, 0, 255]
            except IndexError:
                pass

        if all(particle.position in self._agent.target_area for particle in self._swarm):
            self._is_done = True
        
        return percept['view']
