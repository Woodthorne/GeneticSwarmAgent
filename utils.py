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
    if all(array == [100, 100, 100]):
        return ColorEnum.WHITE
    raise ValueError(f'Unrecognised colour: {array}')


def euclidean(origin: np.ndarray, destination: np.ndarray) -> int:
    return np.linalg.norm(origin-destination)


### UNUSED ###

@cache
def add(*vectors: tuple[float]) -> tuple[float]:
    '''Adds together multiple vectors

    Parameters
    ----------
    vectors: tuple[*float]
        Vectors to be added.

    Returns
    -------
    tuple[*float]
        Vector sum.
    '''
    assert all(len(vectors[0]) == len(vector) for vector in vectors), \
        'All vectors need to be of same length.'
    vector = tuple(int(sum(vals)) for vals in zip(*vectors))
    return vector


@cache
def subtract(vector: tuple[float], *subtrahends: tuple[float]) -> tuple[float]:
    '''Subtracts vectors from first vectors.

    Parameters
    ----------
    vector: tuple[*float]
        Vector to subtract from.
    subtrahends: tuple[*float]
        Vectors to subtract from first vector.
    
    Returns
    -------
    tuple[*float]
        Vector difference.
    '''
    assert all(len(vector) == len(subtrahend) for subtrahend in subtrahends), \
        'All subtrahends need to be of same length as vector.'
    addends = [multiply(-1, subtrahend) for subtrahend in subtrahends]
    return add(vector, *addends)


@cache
def multiply(scalar: float, vector: tuple[float]) -> tuple[float]:
    '''Scalar multiplication
    
    Parameters
    ----------
    scalar: float
        Left scalar of multiplication
    vector: tuple[*float]
        Vector to be multiplied
    
    Returns
    -------
    tuple[*float]
        Scaled vector.

    '''
    return tuple(int(scalar * val) for val in vector)
