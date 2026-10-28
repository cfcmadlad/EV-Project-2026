from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.balancing import simulate
from src.config import FloatArray, ProjectConfig, Technique
from src.dataset import draw_block, pack_features
from src.pack import PackBatch

GRID = 64
CHUNK = 250
N_CELLS = 4
TARGET_SPREAD = 0.01
OUTPUT_CHANNELS = 1 + N_CELLS


@dataclass(frozen=True, slots=True)
class Trajectories:
    features: FloatArray
    technique: np.ndarray
    time_s: FloatArray
    curves: FloatArray


def _slice(packs: PackBatch, rows: slice) -> PackBatch:
    return PackBatch(packs.capacity_ah[rows], packs.r0_ohm[rows], packs.soc0[rows])


def trajectory_dataset(config: ProjectConfig, rng: np.random.Generator, runs: int) -> Trajectories:
    features, technique, times, curves = [], [], [], []
    grid = np.linspace(0.0, 1.0, GRID)
    for tech in Technique:
        packs = draw_block(tech, config, rng, runs)
        component, frequency = packs[1], packs[2]
        for first in range(0, runs, CHUNK):
            rows = slice(first, first + CHUNK)
            block = _slice(packs[0], rows)
            result = simulate(tech, block, component[rows], frequency[rows], config, trace=True)
            for k in range(block.n_packs):
                if not result.balanced[k]:
                    continue
                t_end = result.balancing_time_s[k]
                soc = np.stack(
                    [
                        np.interp(grid * t_end, result.trace_time_s, result.trace_soc[:, k, c])
                        for c in range(N_CELLS)
                    ],
                    axis=1,
                )
                curves.append(np.column_stack((np.ptp(soc, axis=1), soc)))
                times.append(t_end)
                technique.append(int(tech))
                features.append(pack_features(tech, block, component[rows], frequency[rows])[k])
    return Trajectories(
        np.array(features),
        np.array(technique),
        np.array(times),
        np.array(curves, dtype=np.float64),
    )


class GruDecoder(nn.Module):
    def __init__(self, n_features: int, width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.gru = nn.GRU(width + 1, width, batch_first=True)
        self.out = nn.Linear(width, OUTPUT_CHANNELS)
        self.time = nn.Linear(width, 1)
        self.register_buffer("grid", torch.linspace(0.0, 1.0, GRID))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(features)
        steps = self.grid.expand(features.shape[0], GRID).unsqueeze(2)
        inputs = torch.cat((steps, context.unsqueeze(1).expand(-1, GRID, -1)), dim=2)
        hidden, _ = self.gru(inputs, context.unsqueeze(0).contiguous())
        return self.out(hidden), self.time(context).squeeze(1)


class TransformerDecoder(nn.Module):
    def __init__(self, n_features: int, width: int) -> None:
        super().__init__()
        dim = width // 2
        self.encoder = nn.Sequential(nn.Linear(n_features, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.query = nn.Parameter(torch.randn(1, GRID, dim) * 0.1)
        self.step = nn.Linear(1, dim)
        layer = nn.TransformerDecoderLayer(
            dim, 2, dim_feedforward=2 * dim, dropout=0.0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.out = nn.Linear(dim, OUTPUT_CHANNELS)
        self.time = nn.Linear(dim, 1)
        self.register_buffer("grid", torch.linspace(0.0, 1.0, GRID))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(features)
        queries = self.query + self.step(self.grid.view(1, GRID, 1))
        hidden = self.decoder(queries.expand(features.shape[0], -1, -1), context.unsqueeze(1))
        return self.out(hidden), self.time(context).squeeze(1)


DECODERS = {"gru_decoder": GruDecoder, "transformer_decoder": TransformerDecoder}


def crossing_norm_time(spread: FloatArray, threshold: float) -> FloatArray:
    grid = np.linspace(0.0, 1.0, GRID)
    out = np.ones(spread.shape[0])
    for k, curve in enumerate(spread):
        below = np.nonzero(curve <= threshold)[0]
        if below.size == 0:
            continue
        j = below[0]
        if j == 0:
            out[k] = 0.0
        else:
            frac = (curve[j - 1] - threshold) / (curve[j - 1] - curve[j])
            out[k] = grid[j - 1] + frac * (grid[j] - grid[j - 1])
    return out


def trajectory_scores(
    true_curves: FloatArray,
    pred_curves: FloatArray,
    true_time: FloatArray,
    pred_time: FloatArray,
    technique: np.ndarray,
) -> dict:
    out = {}
    groups = {"overall": np.ones_like(technique, dtype=bool)} | {
        t.key: technique == t for t in Technique
    }
    for name, mask in groups.items():
        spread_t, spread_p = true_curves[mask, :, 0], pred_curves[mask, :, 0]
        cross_true = true_time[mask]
        cross_pred = crossing_norm_time(spread_p, TARGET_SPREAD) * pred_time[mask]
        half_true = np.array([crossing_norm_time(r[None], 0.5 * r[0])[0] for r in spread_t])
        half_pred = np.array([crossing_norm_time(r[None], 0.5 * r[0])[0] for r in spread_p])
        out[name] = {
            "spread_rmse_percent": float(100.0 * np.sqrt(np.mean((spread_t - spread_p) ** 2))),
            "cell_soc_rmse_percent": float(
                100.0 * np.sqrt(np.mean((true_curves[mask, :, 1:] - pred_curves[mask, :, 1:]) ** 2))
            ),
            "time_to_1pct_median_rel_error_percent": float(
                100.0 * np.median(np.abs(cross_pred / cross_true - 1.0))
            ),
            "total_time_median_rel_error_percent": float(
                100.0 * np.median(np.abs(pred_time[mask] / true_time[mask] - 1.0))
            ),
            "half_spread_time_true": float(half_true.mean()),
            "half_spread_time_pred": float(half_pred.mean()),
        }
    return out
