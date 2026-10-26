from dataclasses import dataclass, field

import numpy as np

from src.config import PERCENT, FloatArray
from src.params import SocEstimationSpec
from src.soc_estimation.coulomb import coulomb_count
from src.soc_estimation.data import Cycle
from src.soc_estimation.ekf import EkfCell, run_ekf


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    offset_a: float
    noise_v: float
    wrong_start: bool


def scenarios(spec: SocEstimationSpec) -> dict[str, Scenario]:
    small = spec.tester_accuracy_fraction * spec.tester_full_scale_a
    large = spec.injected_offset_fraction * spec.tester_full_scale_a
    return {
        "nominal": Scenario("nominal", 0.0, 0.0, False),
        "wrong_start": Scenario("wrong_start", 0.0, 0.0, True),
        "noise_5mv": Scenario("noise_5mv", 0.0, spec.voltage_noise_v, False),
        "offset_tester_accuracy": Scenario("offset_tester_accuracy", small, 0.0, False),
        "offset_injected": Scenario("offset_injected", large, 0.0, False),
        "wrong_start_offset_injected": Scenario("wrong_start_offset_injected", large, 0.0, True),
    }


@dataclass(slots=True)
class ErrorSums:
    n: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    max_abs: float = 0.0
    start_errors: list[float] = field(default_factory=list)
    per_cycle_rmse: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, tail: FloatArray, first_error: float) -> None:
        self.n += tail.size
        self.sum_sq += float(np.sum(tail**2))
        self.sum_abs += float(np.sum(np.abs(tail)))
        self.max_abs = max(self.max_abs, float(np.abs(tail).max()))
        self.start_errors.append(abs(first_error))
        self.per_cycle_rmse[name] = float(np.sqrt(np.mean(tail**2))) * PERCENT

    def leaf(self) -> dict[str, object]:
        return {
            "rmse_percent": float(np.sqrt(self.sum_sq / self.n)) * PERCENT,
            "mae_percent": self.sum_abs / self.n * PERCENT,
            "max_abs_percent": self.max_abs * PERCENT,
            "abs_error_at_300s_percent_mean": float(np.mean(self.start_errors)) * PERCENT,
            "abs_error_at_300s_percent_max": float(np.max(self.start_errors)) * PERCENT,
            "n_samples": self.n,
            "n_cycles": len(self.start_errors),
            "per_cycle_rmse_percent": dict(sorted(self.per_cycle_rmse.items())),
        }


def start_index(spec: SocEstimationSpec) -> int:
    return round(spec.wrong_start_check_s / spec.step_s) - 1


def perturbed_inputs(
    cycle: Cycle, scenario: Scenario, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    current = cycle.current_a + scenario.offset_a
    noise = rng.standard_normal(cycle.n_steps) * scenario.noise_v
    return current, cycle.voltage_v + noise


def initial_soc(cycle: Cycle, scenario: Scenario, spec: SocEstimationSpec) -> float:
    return 1.0 - (spec.wrong_start_soc if scenario.wrong_start else 0.0)


def cycle_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, index]))


def run_coulomb(cycle: Cycle, scenario: Scenario, spec: SocEstimationSpec) -> FloatArray:
    return coulomb_count(
        cycle.current_a,
        initial_soc(cycle, scenario, spec),
        cycle.capacity_ah,
        spec.step_s,
        scenario.offset_a,
    )


def run_ekf_scenario(
    cell: EkfCell,
    cycle: Cycle,
    scenario: Scenario,
    process_std: float,
    measurement_std_v: float,
    spec: SocEstimationSpec,
    rng: np.random.Generator,
) -> FloatArray:
    current, voltage = perturbed_inputs(cycle, scenario, rng)
    return run_ekf(
        cell,
        current,
        voltage,
        initial_soc(cycle, scenario, spec),
        process_std,
        measurement_std_v,
        spec.ekf_initial_std,
        spec.step_s,
    )
