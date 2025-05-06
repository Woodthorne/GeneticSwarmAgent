from functools import cache
from typing import Iterator

import numpy as np

from signals import ColorEnum


@cache
def add(*vectors: list[int]) -> list[int]:
    assert all(len(vectors[0]) == len(vector) for vector in vectors), \
        'All vectors need to be of same length.'
    
    vector = [sum(vals) for vals in zip(*vectors)]
    return vector


def drill(array: np.ndarray) -> Iterator[tuple[list[int], np.ndarray]]:
    subarray: np.ndarray
    for index, subarray in enumerate(array):
        if len(subarray.shape) == 1:
            yield ([index], subarray)
        else:
            for result in drill(subarray):
                result[0].insert(0, index)
                yield result


def get_color(array: np.ndarray) -> ColorEnum:
    if all(array == [255, 255, 255]):
        return ColorEnum.WHITE
    if all(array == [000, 000, 000]):
        return ColorEnum.BLACK
    if all(array == [000, 000, 255]):
        return ColorEnum.RED
    if all(array == [000, 255, 000]):
        return ColorEnum.GREEN
    if all(array == [255, 000, 000]):
        return ColorEnum.BLUE
    raise ValueError(f'Unrecognised colour: {array}')
