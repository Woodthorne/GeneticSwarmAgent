import cv2
import numpy as np

from agent import Agent
from environment import Environment
from map import Map
from signals import ColorEnum


def main(map_name: str, swarm_size: int = 10) -> None:
    map_ = Map(map_name = map_name)
    agent = Agent(
        map_shape = map_.axes,
        target_area = map_.find_positions(ColorEnum.GREEN)
    )
    environment = Environment(
        map = map_,
        agent = agent,
        sensor_radius = 10,
        collisions = False
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