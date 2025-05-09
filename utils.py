import math
from functools import cache
from typing import Iterator

import numpy as np

from signals import ColorEnum


type Point = tuple[float, ...]
type Vector = tuple[Point, Point]


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


def inside_zone(point: Point, zone: Vector) -> bool:
    for ax0, ax1, axp in zip(*zone, point):
        if not ax0 < ax1 < axp:
            return False
    return True


def sightline(origin: Point, radius: float, angle: float) -> np.ndarray:
    assert len(origin) == 2, 'Only implemented for 2 dimensions'
    radians = math.radians(angle)
    mod = np.array(
        [radius * math.cos(radians),
         radius * math.sin(radians)]
    )

    vector = np.array([origin, origin + mod])
    return vector


def intersection(vector_a: np.ndarray, vector_b: np.ndarray) -> tuple[bool, np.ndarray|None]:
    (x1, y1), (x2, y2) = vector_a
    (x3, y3), (x4, y4) = vector_b
    if (x1 == x2 and x3 == x4) or (y1 == y2 and y3 == y4):
        return (False, None)
    if np.unique(vector_a, axis=0).shape != vector_a.shape:
        return (False, None)
    
    A = np.array([[x2 - x1, -(x4 - x3)], [y2 - y1, -(y4 - y3)]])
    b = np.array([x3 - x1, y3 - y1])
    x = np.linalg.solve(A, b)
    
    intersect = np.all(0 <= x) and np.all(x <= 1)
    if intersect:
        coordinate = np.array([x1 + x[0] * (x2 - x1),
                               y1 + x[0] * (y2 - y1)])
        return (True, coordinate)
    else:
        return (False, None)

    # https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_lines
    # https://numpy.org/doc/2.2/reference/generated/numpy.linalg.solve.html


def collision(vector: Vector, point: Point) -> bool:
    x0, y0 = point
    (x1, y1), (x2, y2) = vector
    if sorted([y0, y1, y2])[1] != y0:
        return False
    
    dx = abs(x1 - x2)
    if dx == 0:
        return x0 == x1
    
    else:
        dy = abs(y1 - y2)
        c = dy / dx
        m = (y1 + c) / x1
        return (m * x0) + c == y0


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
