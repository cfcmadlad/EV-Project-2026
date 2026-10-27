from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.config import FloatArray
from src.soc_estimation.data import Cycle

INPUTS = 3


@dataclass(frozen=True, slots=True)
class Normaliser:
    mean: tuple[float, ...]
    std: tuple[float, ...]


class SocGru(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(INPUTS, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(window)
        return self.head(output[:, -1]).squeeze(-1)


def parameter_count(hidden_size: int) -> int:
    return 3 * (INPUTS * hidden_size + hidden_size**2 + 2 * hidden_size) + hidden_size + 1


def macs_per_estimate(hidden_size: int, window_s: int) -> int:
    per_step = 3 * (INPUTS * hidden_size + hidden_size**2)
    return window_s * per_step + hidden_size


def feature_matrix(
    voltage_v: FloatArray, current_a: FloatArray, temperature_c: FloatArray
) -> FloatArray:
    return np.stack((voltage_v, current_a, temperature_c), axis=1)


def cycle_features(cycle: Cycle) -> FloatArray:
    return feature_matrix(cycle.voltage_v, cycle.current_a, cycle.case_temperature_c)


def fit_normaliser(cycles: list[Cycle]) -> Normaliser:
    stacked = np.concatenate([cycle_features(c) for c in cycles])
    return Normaliser(tuple(stacked.mean(axis=0).tolist()), tuple(stacked.std(axis=0).tolist()))


def normalise(features: FloatArray, normaliser: Normaliser) -> torch.Tensor:
    scaled = (features - np.array(normaliser.mean)) / np.array(normaliser.std)
    return torch.tensor(scaled, dtype=torch.float32)


def window_ends(n_steps: int, window_s: int, stride: int) -> np.ndarray:
    return np.arange(window_s - 1, n_steps, stride)


def gather(features: torch.Tensor, ends: np.ndarray, window_s: int) -> torch.Tensor:
    index = torch.as_tensor(ends[:, None] - (window_s - 1) + np.arange(window_s)[None, :])
    return features[index]


@torch.no_grad()
def predict(
    model: SocGru,
    features: torch.Tensor,
    ends: np.ndarray,
    window_s: int,
    batch_size: int = 1024,
) -> FloatArray:
    model.eval()
    chunks = [
        model(gather(features, ends[start : start + batch_size], window_s)).numpy()
        for start in range(0, ends.size, batch_size)
    ]
    return np.concatenate(chunks).astype(np.float64)
