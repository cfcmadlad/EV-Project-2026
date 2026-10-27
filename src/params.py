from dataclasses import dataclass
from enum import StrEnum

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


class FlyingCapModel(StrEnum):
    MIDSEM = "midsem"
    TWO_PHASE = "two_phase"


@dataclass(frozen=True, slots=True)
class FlyingCapSpec:
    component: ComponentRange = ComponentRange(1e-3, 10e-3)
    frequency_hz: ComponentRange = ComponentRange(1e3, 10e3)
    r_ds_on_ohm: float = 0.01
    esr_ohm: float = 0.005
    model: FlyingCapModel = FlyingCapModel.TWO_PHASE


TechniqueSpec = PassiveSpec | BuckBoostSpec | FlyingCapSpec


class LinkCoupling(StrEnum):
    PER_LINK = "per_link"
    NET_CURRENT = "net_current"


@dataclass(frozen=True, slots=True)
class SimSpec:
    dt_s: float = 1.0
    t_max_s: float = 48.0 * SECONDS_PER_HOUR
    soc_tolerance: float = 0.01
    link_soc_deadband: float = 0.002
    link_coupling: LinkCoupling = LinkCoupling.PER_LINK
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
    coupling_packs: int = 100
    dt_scaling_steps_s: tuple[float, ...] = (1.0, 0.5, 0.25)
    passive_closed_form_tolerance: float = 0.03
    charge_conservation_tolerance: float = 1e-9


@dataclass(frozen=True, slots=True)
class IdentificationSpec:
    ocv_points: int = 101
    pulse_current_a: float = 3.0
    pulse_current_tolerance_a: float = 0.15
    pulse_max_samples: int = 110
    tau_bounds_s: tuple[float, float] = (1.0, 300.0)
    r1_bounds_ohm: tuple[float, float] = (1e-5, 0.5)
    fast_tau_bounds_s: tuple[float, float] = (0.1, 30.0)
    slow_tau_bounds_s: tuple[float, float] = (30.0, 3000.0)
    bound_flag_tolerance: float = 1e-3
    constant_window: tuple[float, float] = (0.30, 0.70)
    validation_window: tuple[float, float] = (0.20, 0.80)
    rmse_whole_cycle_v: float = 0.030
    rmse_window_v: float = 0.020
    bootstrap_resamples: int = 500
    bootstrap_block_samples: int = 50
    slope_window: tuple[float, float] = (0.45, 0.55)
    slope_block_points: int = 5
    ci_percent: float = 95.0
    rest_noise_samples: int = 90


@dataclass(frozen=True, slots=True)
class DriveCycleSpec:
    lower_voltage_v: float = 2.8
    upper_voltage_v: float = 4.2
    charge_c_rate: float = 0.5
    max_cycle_repeats: int = 20
    charge_horizon_s: int = 10800
    population_packs: int = 200
    primary_cycle: str = "Mixed1"
    profile_step_s: float = 1.0
    trace_interval_s: float = 10.0
    soc_conservation_tolerance: float = 1e-9
    ledger_tolerance: float = 0.01


@dataclass(frozen=True, slots=True)
class SocErrorSpec:
    sigma_percent: tuple[float, ...] = (0.0, 1.0, 3.0)
    packs: int = 200
    voltage_error_mv: float = 10.0
    spread_tail_percent: float = 95.0
    max_balancing_h: float = 12.0
    deterministic_check_packs: int = 20


@dataclass(frozen=True, slots=True)
class ConverterCheckSpec:
    soc_pairs: tuple[tuple[float, float], ...] = ((0.55, 0.50), (0.51, 0.50))
    buck_boost_points: tuple[tuple[float, float], ...] = (
        (10e-6, 50e3),
        (55e-6, 30e3),
        (100e-6, 10e3),
    )
    flying_capacitor_points: tuple[tuple[float, float], ...] = (
        (1e-3, 1e3),
        (5.5e-3, 5.5e3),
        (10e-3, 10e3),
    )
    max_step_divisor: int = 400
    refined_step_divisor: int = 800
    settle_periods: int = 300
    measured_periods: int = 60
    stationarity_tolerance: float = 1e-3
    current_tolerance: float = 0.05
    loss_tolerance: float = 0.15
    switch_roff_ohm: float = 1e6
    gate_edge_fraction: float = 1e-3
    relative_tolerance: float = 1e-4
    dcm_idle_fraction_min: float = 0.05
    energy_balance_tolerance: float = 0.005
    step_independence_tolerance: float = 0.005


@dataclass(frozen=True, slots=True)
class MLEvalSpec:
    n_folds: int = 5
    oor_component_quantile: float = 0.80
    oor_spread_limit: float = 0.08
    conformal_fit_fraction: float = 0.75
    conformal_level: float = 0.90
    coverage_low: float = 0.87
    coverage_high: float = 0.93
    conformal_repeats: int = 10
    passive_tolerance: float = 0.03
    time_grid_points: int = 160
    time_grid_min_s: float = 1.0
    bisection_steps: int = 40
    timing_repeats: int = 5
    simulator_timing_repeats: int = 1
    timing_batch_sizes: tuple[int, ...] = (1, 1000)


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
    flying_capacitor=FlyingCapSpec(model=FlyingCapModel.MIDSEM),
)


@dataclass(frozen=True, slots=True)
class SocEstimationSpec:
    train_temperatures_c: tuple[int, ...] = (0, 25, 40)
    test_b_temperature_c: int = 10
    train_mixed: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    validation_mixed: tuple[int, ...] = (7,)
    validation_mixed_override: tuple[tuple[int, tuple[int, ...]], ...] = ((40, (3,)),)
    test_a_cycles: tuple[str, ...] = ("Mixed8", "UDDS", "HWFET", "LA92", "US06")
    step_s: float = 1.0
    min_discharged_fraction: float = 0.8
    wrong_start_soc: float = 0.20
    wrong_start_check_s: float = 300.0
    tester_full_scale_a: float = 75.0
    tester_accuracy_fraction: float = 0.001
    injected_offset_fraction: float = 0.005
    voltage_noise_v: float = 0.005
    windows_s: tuple[int, ...] = (100, 300)
    hidden_size: int = 32
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 12
    batch_size: int = 256
    train_stride_s: int = 6
    learning_rate: float = 3e-3
    ekf_process_std_grid: tuple[float, ...] = (1e-7, 1e-6, 1e-5, 1e-4)
    ekf_measurement_std_v_grid: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16)
    ekf_initial_std: float = 0.2
    pool_window: tuple[float, float] = (0.3, 0.7)
    feedthrough_packs: int = 200


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    calibration_cycles: tuple[str, ...] = ("HWFET", "US06", "Mixed1", "Mixed2")
    validation_cycles: tuple[str, ...] = ("UDDS", "LA92")
    step_s: float = 1.0


@dataclass(frozen=True, slots=True)
class ScalingSpec:
    cells: tuple[int, ...] = (4, 8, 16, 32, 96)
    packs: int = 200
    max_balancing_h: float = 1500.0
    deadband_fraction: float = 0.9
    large_n_min: int = 8
    slope_tolerance: float = 0.25


@dataclass(frozen=True, slots=True)
class RlSpec:
    control_step_s: int = 10
    d_max: float = 0.4
    dcm_margin: float = 0.95
    estimate_sigma: float = 0.01
    loss_weight_ah_per_j: float = 1e-4
    train_cycles: tuple[str, ...] = ("Mixed1", "Mixed2", "Mixed3", "Mixed4", "Mixed5", "Mixed6")
    test_cycles: tuple[str, ...] = ("Mixed7", "Mixed8", "UDDS", "US06")
    test_packs: int = 100
    seeds: tuple[int, ...] = (0, 1, 2)
    total_timesteps: int = 300_000
    gamma: float = 0.999
    gae_lambda: float = 0.98
    log_std_init: float = -1.0
    n_steps: int = 1024
    batch_size: int = 256
    learning_rate: float = 3e-4
    hidden_units: int = 64
    elapsed_reference_s: float = 7200.0
    current_window_s: int = 60
    standard_errors: float = 2.0
