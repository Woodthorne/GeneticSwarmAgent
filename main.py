import cv2
import numpy as np

from agent import Agent
from environment import Environment
from map import AbstractMap, ImgMap
from signals import ColorEnum


def main(map_name: str = None, swarm_size: int = 10, collisions: bool = True) -> None:
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
                ((20, 101), (130, 99))
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
        sensor_radius = 10,
        collisions = collisions
    )
    environment.populate_swarm(swarm_size)
    while not environment.is_done:
        new_frame = environment.step()
        cv2.imshow('frame', new_frame.astype(np.uint8))
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    map_name = 'sample.png'
    main(map_name)