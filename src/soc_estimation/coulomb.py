import numpy as np

from src.config import FloatArray
from src.params import SECONDS_PER_HOUR


def coulomb_count(
    current_a: FloatArray, soc0: float, capacity_ah: float, step_s: float, offset_a: float
) -> FloatArray:
    charge_ah = np.cumsum((current_a + offset_a) * step_s) / SECONDS_PER_HOUR
    return soc0 - charge_ah / capacity_ah
