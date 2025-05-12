from timeit import default_timer

import cv2
import numpy as np

from agent import Agent
from environment import Environment
from map import AbstractMap, ImgMap
from signals import ColorEnum


def main(map_name: str = None, swarm_size: int = 10, collisions: bool = True, sensor_radius: int = 10) -> None:
    if map_name:
        map_ = ImgMap(map_name = map_name)
        agent = Agent(
            map_shape = map_.axes,
            target_area = map_.find_positions(ColorEnum.GREEN),
            map_type = type(map_)
        )
    else:
        map_ = AbstractMap(
            axes = (150, 200),
            start_zone = ((10, 10), (50, 50)),
            goal_zone = ((100, 150), (140, 190)),
            obstacles = [
                ((20, 100), (130, 100))
            ]
        )
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

    environment.populate_swarm(swarm_size)

    cycles = 0
    while not environment.is_done:
        cycles += 1
        print('Running cycle', cycles)
        cycle_start = default_timer()
        new_frame = environment.step()
        print(f'Cycle {cycles} complete. Elapsed time: {round(default_timer() - cycle_start, 3)} s')
        cv2.imshow('frame', new_frame.astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # map_name = 'sample.png'
    map_name = None
    main(map_name, collisions=False, sensor_radius=50)