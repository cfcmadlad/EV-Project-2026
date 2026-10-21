from dataclasses import dataclass

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class ComponentRange:
    low: float
    high: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.low + self.high)


@dataclass(frozen=True, slots=True)
class OCVTable:
    soc: tuple[float, ...]
    volt_v: tuple[float, ...]
    grid_points: int = 2001


@dataclass(frozen=True, slots=True)
class CellParams:
    capacity_ah: float
    r0_ohm: float
    r1_ohm: float
    c1_farad: float
    ocv: OCVTable

    @property
    def tau_s(self) -> float:
        return self.r1_ohm * self.c1_farad


@dataclass(frozen=True, slots=True)
class PackSpec:
    n_cells: int = 4
    capacity_tolerance: float = 0.05
    r0_tolerance: float = 0.10
    soc_center: float = 0.50
    soc_spread_min: float = 0.02
    soc_spread_max: float = 0.10


@dataclass(frozen=True, slots=True)
class PassiveSpec:
    component: ComponentRange = ComponentRange(20.0, 100.0)
    frequency_hz: ComponentRange = ComponentRange(0.0, 0.0)


@dataclass(frozen=True, slots=True)
class BuckBoostSpec:
    component: ComponentRange = ComponentRange(10e-6, 100e-6)
    frequency_hz: ComponentRange = ComponentRange(10e3, 50e3)
    duty: float = 0.4
    r_inductor_ohm: float = 0.08
    r_ds_on_ohm: float = 0.02


@dataclass(frozen=True, slots=True)
class FlyingCapSpec:
    component: ComponentRange = ComponentRange(1e-3, 10e-3)
    frequency_hz: ComponentRange = ComponentRange(1e3, 10e3)
    r_ds_on_ohm: float = 0.01
    esr_ohm: float = 0.005


TechniqueSpec = PassiveSpec | BuckBoostSpec | FlyingCapSpec


@dataclass(frozen=True, slots=True)
class SimSpec:
    dt_s: float = 1.0
    t_max_s: float = 48.0 * SECONDS_PER_HOUR
    soc_tolerance: float = 0.01
    link_soc_deadband: float = 0.002
    trace_interval_s: float = 10.0


@dataclass(frozen=True, slots=True)
class SweepSpec:
    n_points: int = 17


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    runs_per_technique: int = 2000


@dataclass(frozen=True, slots=True)
class MLSpec:
    test_fraction: float = 0.2
    max_iter: int = 500
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    permutation_repeats: int = 10
    timing_repeats: int = 20


@dataclass(frozen=True, slots=True)
class SanitySpec:
    min_balancing_time_s: float = 60.0
    slowdown_window: float = 0.02
    min_capacitor_slowdown: float = 3.0


@dataclass(frozen=True, slots=True)
class CheckSpec:
    packs_per_technique: int = 200
    ledger_tolerance: float = 0.01
    dt_scaling_packs: int = 20
    dt_scaling_steps_s: tuple[float, ...] = (1.0, 0.5, 0.25)
    passive_closed_form_tolerance: float = 0.03
    charge_conservation_tolerance: float = 1e-9


@dataclass(frozen=True, slots=True)
class ParameterSet:
    name: str
    cell: CellParams
    pack: PackSpec
    passive: PassiveSpec
    buck_boost: BuckBoostSpec
    flying_capacitor: FlyingCapSpec


ASSUMED_OCV = OCVTable(
    soc=(
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ),
    volt_v=(
        3.00,
        3.35,
        3.45,
        3.51,
        3.55,
        3.58,
        3.61,
        3.63,
        3.65,
        3.68,
        3.71,
        3.74,
        3.78,
        3.82,
        3.87,
        3.92,
        3.97,
        4.02,
        4.07,
        4.13,
        4.20,
    ),
)

ASSUMED = ParameterSet(
    name="assumed",
    cell=CellParams(
        capacity_ah=3.0,
        r0_ohm=0.030,
        r1_ohm=0.015,
        c1_farad=2000.0,
        ocv=ASSUMED_OCV,
    ),
    pack=PackSpec(),
    passive=PassiveSpec(),
    buck_boost=BuckBoostSpec(),
    flying_capacitor=FlyingCapSpec(),
)
