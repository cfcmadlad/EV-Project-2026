from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.fitted import FITTED_HG2_25C
from src.params import (
    BuckBoostSpec,
    CellParams,
    CheckSpec,
    ConverterCheckSpec,
    DatasetSpec,
    FlyingCapSpec,
    IdentificationSpec,
    MLSpec,
    PackSpec,
    ParameterSet,
    PassiveSpec,
    SanitySpec,
    SimSpec,
    SweepSpec,
    TechniqueSpec,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    seed: int = 2026
    params: ParameterSet = FITTED_HG2_25C
    sim: SimSpec = SimSpec()
    sweep: SweepSpec = SweepSpec()
    dataset: DatasetSpec = DatasetSpec()
    ml: MLSpec = MLSpec()
    sanity: SanitySpec = SanitySpec()
    checks: CheckSpec = CheckSpec()
    identification: IdentificationSpec = IdentificationSpec()
    converter_check: ConverterCheckSpec = ConverterCheckSpec()
    plot: PlotConfig = PlotConfig()

    @property
    def cell(self) -> CellParams:
        return self.params.cell

    @property
    def pack(self) -> PackSpec:
        return self.params.pack

    @property
    def passive(self) -> PassiveSpec:
        return self.params.passive

    @property
    def buck_boost(self) -> BuckBoostSpec:
        return self.params.buck_boost

    @property
    def flying_capacitor(self) -> FlyingCapSpec:
        return self.params.flying_capacitor

    def technique(self, technique: Technique) -> TechniqueSpec:
        return (self.passive, self.buck_boost, self.flying_capacitor)[technique]


CONFIG = ProjectConfig()
