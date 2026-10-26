import numpy as np

from src.config import FloatArray

BIN_EPSILON = 1e-9


def interval_mean(time_s: FloatArray, value: FloatArray, step_s: float) -> FloatArray:
    area = np.concatenate(([0.0], np.cumsum(0.5 * (value[1:] + value[:-1]) * np.diff(time_s))))
    n_bins = int(np.floor((time_s[-1] - time_s[0]) / step_s + BIN_EPSILON))
    edges = time_s[0] + step_s * np.arange(n_bins + 1)
    return np.diff(np.interp(edges, time_s, area)) / step_s


def interval_end(time_s: FloatArray, value: FloatArray, step_s: float) -> FloatArray:
    n_bins = int(np.floor((time_s[-1] - time_s[0]) / step_s + BIN_EPSILON))
    edges = time_s[0] + step_s * np.arange(1, n_bins + 1)
    return np.interp(edges, time_s, value)
