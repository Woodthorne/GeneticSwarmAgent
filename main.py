from timeit import default_timer

import cv2
import numpy as np

from agent import Agent
from drone import Drone
from environment import Environment
from map import AbstractMap, ImgMap
from signals import ColorEnum


def main(
        map_params: dict,
        swarm_size: int = 10,
        collisions: bool = True,
        sensor_radius: int = 10
) -> None:
    # if map_name:
    #     map_ = ImgMap(map_name = map_name)
    #     agent = Agent(
    #         map_shape = map_.axes,
    #         target_area = map_.find_positions(ColorEnum.GREEN),
    #         map_type = type(map_)
    #     )
    # else:
    map_ = AbstractMap(**map_params)
    
    agent = Agent(
        map_shape = map_.axes,
        target_area = map_.goal,
        map_type = type(map_)
    )

    environment = Environment(
        map = map_,
        agent = agent,
        sensor_radius = sensor_radius,
        collisions = collisions
    )

    # for num in range(10):
    #     drone = Drone(
    #         id_num=999 - num,
    #         position=np.array((1 + num, 1 + num)),
    #         velocity=np.array((0,0)),
    #         fitness_func=np.sum
    #     )
    #     environment._swarm.append(drone)
    
    environment.populate_swarm(swarm_size)
    
    cycles = 0
    while not environment.is_done:
        cycles += 1
        print('Running cycle', cycles)
        cycle_start = default_timer()
        new_frame = environment.step()
        cycle_time = round(default_timer() - cycle_start, 3)
        print(f'Cycle {cycles} complete. Elapsed time: {cycle_time} s')
        cv2.imshow('frame', new_frame.astype(np.uint8))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    if environment.is_done:
        print('Simulation complete. Press Q to close.')
        cv2.waitKey(0) & 0xFF == ord('q')
    


if __name__ == '__main__':
    map_name = 'sample.png'
    
    abstract_map_params = dict(
        axes = (150, 200),
        start_zone = ((10, 10), (50, 50)),
        goal_zone = ((100, 150), (140, 190)),
        obstacles = [
            ((20, 100), (130, 100))
        ]   
    )
    # map_name = None
    main(map_params=abstract_map_params,
         collisions=False,
         sensor_radius=50)