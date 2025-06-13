import math
from functools import cache
from typing import Iterator

import numpy as np

from signals import ColorEnum

type Vector = np.ndarray
type Segment = np.ndarray


type LPoint = tuple[float, ...]
type LVector = tuple[LPoint, LPoint]


def dev_print(*values, sep = None, end = None, file = None, flush = None) -> None:
    print('DEV', *values, sep=sep, end=end, file=file, flush=flush)


def euclidean(origin: Vector, destination: Vector) -> float:
    distance = np.linalg.norm(origin-destination)
    return float(distance)


def inside_zone(point: Vector, zone: Segment) -> bool:
    for ax0, ax1, axp in zip(*zone, point):
        if not ax0 < axp < ax1:
            return False
    dev_print(point, zone.flatten())
    return True


def intersection(segment_a: Segment, segment_b: Segment) -> tuple[bool, np.ndarray|None]:
    (x1, y1), (x2, y2) = segment_a
    (x3, y3), (x4, y4) = segment_b
    if (x1 == x2 and x3 == x4) or (y1 == y2 and y3 == y4):
        return (False, None)
    if np.unique(segment_a, axis=0).shape != segment_a.shape:
        return (False, None)
    
    # dev_print(segment_a.flatten())
    # dev_print(segment_b.flatten())
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


def get_discrete_coords(segment: Segment) -> list[np.ndarray]:
    (x1, y1), (x2, y2) = segment
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    match x1 == x2, y1 == y2:
        case True, True:
            return [np.array([x1, y1])]
        case False, False:
            dy = y1 - y2
            dx = x1 - x2
            m = dy / dx
            y0 = y1 - m * x1
            
            def f(x: float) -> int:
                return int(m * x + y0)
            
            min_x, max_x = sorted([x1, x2])
            return [np.array([x, f(x)]) for x in range(min_x, max_x + 1)]

        case True, False:
            min_y, max_y = sorted([y1, y2])
            return [np.array([x1, y]) for y in range(min_y, max_y + 1)]
        
        case False, True:
            min_x, max_x = sorted([x1, x2])
            return [np.array([x, y1]) for x in range(min_x, max_x + 1)]
    

def segment_circle_intersection(segment: Segment, center: Vector, radius: float) -> tuple[int, np.ndarray|None]:
    (x1, y1), (x2, y2) = sorted(segment, key=lambda v: v[0])
    a = (y1 - y2)
    b = (x2 - x1)
    c = x2 * y1 - x1 * y2

    # dev_print(f'{radius=}')
    
    delta = radius**2 * (a**2 + b**2) + c**2
    if delta < 0:
        return (0, None)
    
    x0, y0 = center
    # ix1 = x0 + (a*c + b * math.sqrt(delta)) / (a**2 + b**2)
    # ix2 = x0 + (a*c - b * math.sqrt(delta)) / (a**2 + b**2)
    # iy1 = y0 + (b*c - a * math.sqrt(delta)) / (a**2 + b**2)
    # iy2 = y0 + (b*c + a * math.sqrt(delta)) / (a**2 + b**2)
    # i_segment = np.array(((ix1, iy1), (ix2, iy2)))
    # dev_print('c ', i_segment.flatten())
    
    c1 = c - a * x0 - b * y0
    delta = radius**2 * (a**2 + b**2) + c1**2
    ix1 = x0 + (a*c1 + b * math.sqrt(delta)) / (a**2 + b**2)
    ix2 = x0 + (a*c1 - b * math.sqrt(delta)) / (a**2 + b**2)
    iy1 = y0 + (b*c1 - a * math.sqrt(delta)) / (a**2 + b**2)
    iy2 = y0 + (b*c1 + a * math.sqrt(delta)) / (a**2 + b**2)
    i_segment = [[ix1, iy1], [ix2, iy2]]
    # i_segment = np.array(((ix1, iy1), (ix2, iy2)))
    # dev_print('c1', i_segment.flatten())
    # dev_print('found', np.array(i_segment).flatten())
    
    min_x, max_x = x1, x2
    min_y, max_y = sorted([y1, y2])
    y_line = ix1 == ix2
    x_line = iy1 == iy2
    # if not y_line and (max_x < min([ix1, ix2]) or max([ix1, ix2]) < min_x):
    #     return (0, None)
    # if not x_line and (max_y < min([iy1, iy2]) or max([iy1, iy2]) < min_y):
    #     return (0, None)
    # if not y_line:
    #     i_segment.sort(key=lambda v: v[0])
    #     if x_line and (max_x < i_segment[0][0] or i_segment[1][0] < min_x):
    #         dev_print(min_y == max_y == iy1 == iy2)
    #         return (0, None)
    #     if i_segment[0][0] < min_x:
    #         i_segment[0][0] = min_x
    #     if i_segment[1][0] > max_x:
    #         i_segment[1][0] = max_x
    # if not x_line:
    #     i_segment.sort(key=lambda v: v[1])
    #     if y_line and (max_y < i_segment[0][1] or i_segment[1][1] < min_y):
    #         dev_print(min_x == max_x == ix1 == ix2)
    #         return (0, None)
    #     if i_segment[0][1] < min_y:
    #         i_segment[0][1] = min_y
    #     if i_segment[1][1] > max_y:
    #         i_segment[1][1] = max_y
    
    # dev_print('adjusted', np.array(i_segment).flatten())
    # dev_print('segment', segment.flatten())
    # print('##################################')

    if delta > 0:
        i_segment = []
        for x, y in ((ix1, iy1), (ix2, iy2)):
            if x < min_x:
                x = min_x
            if x > max_x:
                x = max_x
            if y < min_y:
                y = min_y
            if y > max_y:
                y = max_y
            i_segment.append((x,y))
        
        # if i_segment[0] == i_segment[1]:
        #     dev_print(segment.flatten())
        #     dev_print(np.array(i_segment).flatten())
        #     quit()
        return (2, np.array(i_segment))
    else:
        dev_print('Possible tangent')
        dev_print('x:', ix1 == ix2)
        dev_print('y:', iy1 == iy2)
        dev_print(np.array(ix1, iy1))
        dev_print(np.array(ix2, iy2))
        quit()
    
    # generalised circle = (x-a)**2 + (y-b)**2 == r**2
    # a, b = center of circle
    # x, y = points on circle

    # standard line = ax + by = c
    # linear = ax + c = -by
    # 
    # (x2 -y1)(y - y1) - (y2 - y1)(x - x1) = 0
    # (y1 - y2)x + (x2 - x1)y + (x1 * y2 - x2 * y1) = 0
    # (y1 - y2)x + (x2 - x1)y = -(x1 * y2 - x2 * y1)
    #
    # a = (y1 - y2)
    # b = (x2 - x1)
    # c = (x1 * y2 - x2 * y1)

    # y − y0 = k(x − x0)
    # (y - y0) / (x - x0) = k


    # TODO: Fix intersections


def merge_segments(segments: list[Segment]) -> list[Segment]:
    # TODO: Clean
    merged = []
    checked = []
    for index, vector_1 in enumerate(segments):
        if index in checked:
            continue
        checked.append(index)
        
        (x1, y1), (x2, y2) = sorted(vector_1, key=lambda p: p[0])
        
        def f(x: float, y: float) -> bool:
            return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) == 0

        merge = False
        for index, vector_2 in enumerate(segments):
            if index in checked:
                continue

            (x3, y3), (x4, y4) = sorted(vector_2, key=lambda p: p[0])
            
            if f(x3, y3) and f(x4, y4):
                if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                    min_x = min([x1, x2, x3, x4])
                    max_x = max([x1, x2, x3, x4])
                    min_y = min([y1, y2, y3, y4])
                    max_y = max([y1, y2, y3, y4])
                    if y1 < y2:
                        new_vector = np.array([[min_x, min_y], [max_x, max_y]])
                    else:
                        new_vector = np.array([[min_x, max_y], [max_x, min_y]])
                    
                    checked.append(index)
                    merged.append(new_vector)
                    merge = True              
        
        if not merge:
            merged.append(vector_1)
    
    if len(segments) != len(merged):
        return merge_segments(merged)
    return merged


def to_tuple(array: np.ndarray) -> tuple:
    try:
        return tuple(to_tuple(item) for item in array)
    except TypeError:
        return array


############################ UNUSED ################################


@cache
def add(*vectors: Vector) -> Vector:
    '''Adds together multiple vectors

    Parameters
    ----------
    vectors: tuple[*float]
        Vectors to be added.

    Returns
    -------
    tuple[*float]
        LVector sum.
    '''
    assert all(len(vectors[0]) == len(vector) for vector in vectors), \
        'All vectors need to be of same length.'
    vector = tuple(int(sum(vals)) for vals in zip(*vectors))
    return vector


def collision(vector: LVector, point: LPoint) -> bool:
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


def intersection_circle(vector: LVector, center: LPoint, radius: float) -> list[np.ndarray]:
    # https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
    x0, y0 = center
    (x1, y1), (x2, y2) = vector
    if x1 == x2:
        if abs(radius) >= abs(x1 - x0):
            p1 = x1, y0 - math.sqrt(radius ** 2 - (x1 - x0) ** 2)
            p2 = x1, y0 + math.sqrt(radius ** 2 - (x1 - x0) ** 2)
            inp = [p for p in (p1, p2)
                   if min(y1, y2) <= p[1] <= max(y1, y2)]
        else:
            inp = []
    
    else:
        k = (y1 / y2) / (x1 / x2)
        b0 = y1 - k * x1
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 +  x0  ** 2 - radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = (-b - math.sqrt(delta)) / (2 * a)
            p2x = (-b + math.sqrt(delta)) / (2 * a)
            p1y = k * x1 + b0
            p2y = k * x2 + b0
            inp = [np.array(p) for p in ((p1x, p1y), (p2x, p2y))
                   if min(x1, x2) <= p[0] <= max(x1, x2)]
        else:
            inp = []

    return inp

    # k * x1 + m = y1
    # k * x2 + m = y2

    # TODO: Field of view circle intercepting vector line

    # line: ax + by = c
    # 

    # circle: x**2 + y**2 = r**2 -> generalised = (x-a)**2 + (y-b)**2 = r**2

    # 2 intersect: r**2(a**2 + b**2) - c**2 > 0
    # 1 intersect: r**2(a**2 + b**2) - c**2 = 0

    # https://en.wikipedia.org/wiki/Linear_equation
    # https://en.wikipedia.org/wiki/Intersection_(geometry)#A_line_and_a_circle


def manhattan(origin: Vector, destination: Vector) -> float:
    distance = 0
    for axo, axd in zip(origin, destination):
        distance += abs(axo - axd)
    return distance


@cache
def multiply(scalar: float, vector: tuple[float]) -> tuple[float]:
    '''Scalar multiplication
    
    Parameters
    ----------
    scalar: float
        Left scalar of multiplication
    vector: tuple[*float]
        LVector to be multiplied
    
    Returns
    -------
    tuple[*float]
        Scaled vector.

    '''
    return tuple(int(scalar * val) for val in vector)


def sightline(origin: LPoint, radius: float, angle: float) -> np.ndarray:
    assert len(origin) == 2, 'Only implemented for 2 dimensions'
    radians = math.radians(angle)
    mod = np.array(
        [radius * math.cos(radians),
         radius * math.sin(radians)]
    )

    vector = np.array([origin, origin + mod])
    return vector


@cache
def subtract(vector: tuple[float], *subtrahends: tuple[float]) -> tuple[float]:
    '''Subtracts vectors from first vectors.

    Parameters
    ----------
    vector: tuple[*float]
        LVector to subtract from.
    subtrahends: tuple[*float]
        Vectors to subtract from first vector.
    
    Returns
    -------
    tuple[*float]
        LVector difference.
    '''
    assert all(len(vector) == len(subtrahend) for subtrahend in subtrahends), \
        'All subtrahends need to be of same length as vector.'
    addends = [multiply(-1, subtrahend) for subtrahend in subtrahends]
    return add(vector, *addends)
