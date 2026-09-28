from dataclasses import dataclass

import numpy as np

from src.config import CellParams, FloatArray, PackConfig


@dataclass(frozen=True)
class PackBatch:
    capacity_ah: FloatArray
    r0_ohm: FloatArray
    soc0: FloatArray

    @property
    def n_packs(self) -> int:
        return self.soc0.shape[0]


def sample_packs(
    n_packs: int,
    rng: np.random.Generator,
    cell: CellParams,
    pack: PackConfig,
    soc_spread: FloatArray | None = None,
) -> PackBatch:
    if n_packs < 1:
        raise ValueError(f"n_packs must be positive, got {n_packs}")
    if soc_spread is None:
        soc_spread = rng.uniform(pack.soc_spread_min, pack.soc_spread_max, n_packs)
    elif soc_spread.shape != (n_packs,) or np.any(
        (soc_spread <= 0.0) | (soc_spread > pack.soc_spread_max)
    ):
        raise ValueError(
            f"soc_spread must have shape ({n_packs},) with values in "
            f"(0, {pack.soc_spread_max}]"
        )
    shape = (n_packs, pack.n_cells)
    capacity = cell.capacity_ah * rng.uniform(
        1.0 - pack.capacity_tolerance, 1.0 + pack.capacity_tolerance, shape
    )
    r0 = cell.r0_ohm * rng.uniform(
        1.0 - pack.r0_tolerance, 1.0 + pack.r0_tolerance, shape
    )
    raw = rng.uniform(size=shape)
    low = raw.min(axis=1, keepdims=True)
    unit = (raw - low) / (raw.max(axis=1, keepdims=True) - low)
    soc0 = pack.soc_center + soc_spread[:, None] * (unit - 0.5)
    return PackBatch(capacity_ah=capacity, r0_ohm=r0, soc0=soc0)


def tile_pack(pack: PackBatch, n_copies: int) -> PackBatch:
    if pack.n_packs != 1 or n_copies < 1:
        raise ValueError("tile_pack expects a single pack and a positive copy count")
    return PackBatch(
        capacity_ah=np.repeat(pack.capacity_ah, n_copies, axis=0),
        r0_ohm=np.repeat(pack.r0_ohm, n_copies, axis=0),
        soc0=np.repeat(pack.soc0, n_copies, axis=0),
    )
