import random
from timeit import default_timer

import cv2
import numpy as np

from agent import Agent
from drone import Drone
from environment import Environment
from map import AbstractMap
from utils import to_tuple


def main(
        map_params: dict,
        swarm_size: int = 10,
        collisions: bool = True,
        sensor_radius: int = 10
) -> None:
    map_ = AbstractMap(**map_params)
    
    agent = Agent(
        target_area = map_.goal,
        step_size = 0.9 * sensor_radius
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
    # drone = Drone(
    #     id_num = 999,
    #     position=np.array((sensor_radius, sensor_radius)),
    #     velocity=np.array((0,0)),
    #     fitness_func=np.sum
    # )
    # environment._swarm.append(drone)
    
    environment.populate_swarm(swarm_size)
    
    cycles = 0
    time_taken = 0
    while not environment.is_done:
        cycles += 1
        print('Running cycle', cycles)
        cycle_start = default_timer()
        new_frame = environment.step()
        cycle_time = round(default_timer() - cycle_start, 3)
        time_taken += cycle_time
        avg_fps = round(cycles / time_taken, 3)
        print(f'Cycle {cycles} complete. Elapsed time: {cycle_time} s ({avg_fps} fps)')
        cv2.imshow('frame', new_frame.astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if environment.is_done:
        print('Simulation complete. Press Q to close.')
        cv2.waitKey(0) & 0xFF == ord('q')
    


if __name__ == '__main__':
    abstract_map_params = dict(
        axes = (300, 400),
        start_zone = ((10, 10), (50, 50)),
        goal_zone = ((250, 350), (290, 390)),
        obstacles = [
            to_tuple(
                [[
                    random.randint(50, 250),
                    random.randint(50, 350)
                ],
                [
                    random.randint(50, 250),
                    random.randint(50, 350)
                ]]
            ) for _ in range(3)
        ]   
    )
    sensor_radius=min(abstract_map_params['axes']) / 10
    main(map_params=abstract_map_params,
         collisions=True,
         sensor_radius=sensor_radius)