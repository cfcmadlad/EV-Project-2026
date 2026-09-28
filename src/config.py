from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SECONDS_PER_HOUR = 3600.0
PERCENT = 100.0
HZ_PER_KHZ = 1e3
MF_PER_FARAD = 1e3

TECHNIQUE_KEYS = ("passive", "buck_boost", "flying_capacitor")
TECHNIQUE_LABELS = ("Passive shunt", "Inductor buck-boost", "Flying capacitor")


class Technique(IntEnum):
    PASSIVE = 0
    BUCK_BOOST = 1
    FLYING_CAPACITOR = 2

    @property
    def key(self) -> str:
        return TECHNIQUE_KEYS[self]

    @property
    def label(self) -> str:
        return TECHNIQUE_LABELS[self]


@dataclass(frozen=True)
class ComponentRange:
    low: float
    high: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.low + self.high)


@dataclass(frozen=True)
class CellParams:
    capacity_ah: float = 3.0
    r0_ohm: float = 0.030
    r1_ohm: float = 0.015
    c1_farad: float = 2000.0
    ocv_soc: tuple[float, ...] = (
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    )  # fmt: skip
    ocv_volt: tuple[float, ...] = (
        3.00, 3.35, 3.45, 3.51, 3.55, 3.58, 3.61, 3.63, 3.65, 3.68, 3.71,
        3.74, 3.78, 3.82, 3.87, 3.92, 3.97, 4.02, 4.07, 4.13, 4.20,
    )  # fmt: skip
    ocv_grid_points: int = 2001


@dataclass(frozen=True)
class PackConfig:
    n_cells: int = 4
    capacity_tolerance: float = 0.05
    r0_tolerance: float = 0.10
    soc_center: float = 0.50
    soc_spread_min: float = 0.02
    soc_spread_max: float = 0.10


@dataclass(frozen=True)
class SimConfig:
    dt_s: float = 1.0
    t_max_s: float = 48.0 * SECONDS_PER_HOUR
    soc_tolerance: float = 0.01
    link_soc_deadband: float = 0.002
    trace_interval_s: float = 10.0


@dataclass(frozen=True)
class PassiveConfig:
    component: ComponentRange = ComponentRange(20.0, 100.0)
    frequency_hz: ComponentRange = ComponentRange(0.0, 0.0)


@dataclass(frozen=True)
class BuckBoostConfig:
    component: ComponentRange = ComponentRange(10e-6, 100e-6)
    frequency_hz: ComponentRange = ComponentRange(10e3, 50e3)
    duty: float = 0.4
    r_inductor_ohm: float = 0.08
    r_ds_on_ohm: float = 0.02


@dataclass(frozen=True)
class FlyingCapacitorConfig:
    component: ComponentRange = ComponentRange(1e-3, 10e-3)
    frequency_hz: ComponentRange = ComponentRange(1e3, 10e3)
    r_ds_on_ohm: float = 0.01
    esr_ohm: float = 0.005


TechniqueConfig: TypeAlias = PassiveConfig | BuckBoostConfig | FlyingCapacitorConfig


@dataclass(frozen=True)
class SweepConfig:
    n_points: int = 17


@dataclass(frozen=True)
class DatasetConfig:
    runs_per_technique: int = 2000


@dataclass(frozen=True)
class ModelConfig:
    test_fraction: float = 0.2
    max_iter: int = 500
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    permutation_repeats: int = 10
    timing_repeats: int = 20


@dataclass(frozen=True)
class SanityConfig:
    min_balancing_time_s: float = 60.0
    slowdown_window: float = 0.02
    min_capacitor_slowdown: float = 3.0


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (8.0, 5.0)
    importance_figsize: tuple[float, float] = (8.0, 7.0)
    font_family: tuple[str, ...] = ("Arial", "Segoe UI", "DejaVu Sans")
    font_size: float = 13.0
    title_size: float = 14.0
    tick_size: float = 11.5
    legend_size: float = 11.0
    line_width: float = 2.0
    grid_line_width: float = 0.8
    reference_line_width: float = 1.2
    legend_anchor: tuple[float, float] = (1.02, 1.0)
    annotation_position: tuple[float, float] = (0.04, 0.94)
    marker_size: float = 14.0
    scatter_alpha: float = 0.45
    band_alpha: float = 0.18
    text_color: str = "#222222"
    grid_color: str = "#e5e5e5"
    band_color: str = "#95a5a6"
    neutral_color: str = "#5d6d7e"
    reference_color: str = "#222222"
    spread_axis_floor: float = 0.5
    log_tick_subs: tuple[float, ...] = (1.0, 2.0, 5.0)
    svg_hash_salt: str = "cell-balancing"
    technique_colors: tuple[str, ...] = ("#c0392b", "#2471a3", "#229954")
    cell_colors: tuple[str, ...] = ("#1f3a5f", "#8e44ad", "#e67e22", "#b7950b")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 2026
    cell: CellParams = CellParams()
    pack: PackConfig = PackConfig()
    sim: SimConfig = SimConfig()
    passive: PassiveConfig = PassiveConfig()
    buck_boost: BuckBoostConfig = BuckBoostConfig()
    flying_capacitor: FlyingCapacitorConfig = FlyingCapacitorConfig()
    sweep: SweepConfig = SweepConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    sanity: SanityConfig = SanityConfig()
    plot: PlotConfig = PlotConfig()

    def technique(self, technique: Technique) -> TechniqueConfig:
        return (self.passive, self.buck_boost, self.flying_capacitor)[technique]


CONFIG = ProjectConfig()
