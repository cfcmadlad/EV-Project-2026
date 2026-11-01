import numpy as np
import pytest

from src.config import CONFIG, Technique
from src.pack import sample_packs
from src.selector import mid_values
from src.thermal import lumped_step, thermal_run


def test_lumped_step_reaches_steady_state_and_balances_energy():
    rise = np.zeros(3)
    power = np.array([0.1, 0.2, 0.3])
    rejected = np.zeros(3)
    for _ in range(2000):
        rise, integral = lumped_step(rise, power, 100.0, 0.05, 1.0)
        rejected += integral / 100.0
    assert rise == pytest.approx(power * 100.0, rel=1e-9)
    stored = 0.05 * rise
    assert stored + rejected == pytest.approx(power * 2000.0, rel=1e-9)


def test_duty_cycling_holds_the_ceiling_and_costs_time():
    rng = np.random.default_rng(3)
    packs = sample_packs(4, rng, CONFIG.cell, CONFIG.pack)
    components, frequencies = mid_values(CONFIG, packs.n_packs)
    passive = Technique.PASSIVE
    free = thermal_run(passive, packs, components[passive], frequencies[passive], CONFIG)
    held = thermal_run(passive, packs, components[passive], frequencies[passive], CONFIG, 10.0)
    tau = CONFIG.thermal.resistor_rth_k_per_w * CONFIG.thermal.resistor_heat_capacity_j_per_k
    one_step = (free.peak_resistor_rise_k - 10.0) * (1.0 - np.exp(-CONFIG.sim.dt_s / tau))
    assert np.all(held.peak_resistor_rise_k <= 10.0 + 1.1 * one_step)
    assert np.all(held.balancing_time_s >= free.balancing_time_s)
    assert held.resistor_heat_j == pytest.approx(held.ledger_resistor_j, rel=1e-9)
