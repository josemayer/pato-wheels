from typing import Tuple

import numpy as np


def get_motor_inner_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    res[shape[0]//2:, :shape[1]//2] = -1
    return res

def get_motor_inner_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    res[shape[0]//2:, shape[1]//2:] = -1
    return res

def get_motor_outer_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    res[shape[0]//2:, :shape[1]//2] = -1
    return res

def get_motor_outer_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    res[shape[0]//2:, shape[1]//2:] = -1
    return res
