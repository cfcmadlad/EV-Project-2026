from dataclasses import replace

import numpy as np

from src.balancing import build_control, simulate
from src.config import CONFIG, Technique
from src.experiments import baseline_pack
from src.params import ASSUMED
from src.stepper import PackState, step

PACK = baseline_pack(CONFIG)


def run_steps(technique: Technique, pack_current_a: np.ndarray | None, n_steps: int):
    component = np.array([CONFIG.technique(technique).component.mid])
    frequency = np.array([CONFIG.technique(technique).frequency_hz.mid])
    control = build_control(technique, PACK, component, frequency, CONFIG)
    state = PackState.initial(PACK.soc0)
    for _ in range(n_steps):
        state = step(state, pack_current_a, control)
    return state


def test_zero_pack_current_equals_rest_step_bitwise() -> None:
    for technique in Technique:
        rest = run_steps(technique, None, 200)
        zero = run_steps(technique, np.zeros(1), 200)
        assert np.array_equal(rest.soc, zero.soc)
        assert np.array_equal(rest.v1_volt, zero.v1_volt)
        assert np.array_equal(rest.removed_j, zero.removed_j)
        assert np.array_equal(rest.element_loss_j, zero.element_loss_j)


def test_stepping_reproduces_the_rest_simulation() -> None:
    config = replace(CONFIG, sim=replace(CONFIG.sim, t_max_s=300.0))
    for technique in Technique:
        component = np.array([config.technique(technique).component.mid])
        frequency = np.array([config.technique(technique).frequency_hz.mid])
        result = simulate(technique, PACK, component, frequency, config)
        state = run_steps(technique, None, 300)
        assert np.array_equal(result.final_soc, state.soc)
        assert np.array_equal(result.energy_removed_j, state.removed_j)


def test_assumed_parameters_still_balance() -> None:
    assumed = replace(CONFIG, params=ASSUMED)
    result = simulate(
        Technique.BUCK_BOOST, baseline_pack(assumed), np.array([55e-6]), np.array([30e3]), assumed
    )
    assert result.balanced.all()
