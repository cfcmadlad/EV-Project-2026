from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from src.balancing import simulate
from src.baselines import (
    capacity_weighted_mean_soc,
    cell_energy_released_j,
    flying_capacitor_prediction,
    minimal_transport_state,
    ocv_slope_v_per_soc,
    passive_prediction,
    physics_prediction,
)
from src.config import CONFIG, Technique
from src.pack import PackBatch, sample_packs
from src.params import SECONDS_PER_HOUR
from src.switched_cap import link_resistance_ohm

N_PACKS = 24


def packs(seed: int = 5) -> PackBatch:
    rng = np.random.default_rng(seed)
    return sample_packs(N_PACKS, rng, CONFIG.cell, CONFIG.pack)


def components(technique: Technique, seed: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hardware = CONFIG.technique(technique)
    return (
        rng.uniform(hardware.component.low, hardware.component.high, N_PACKS),
        rng.uniform(hardware.frequency_hz.low, hardware.frequency_hz.high, N_PACKS),
    )


def test_minimal_transport_conserves_charge_and_fits_the_band() -> None:
    pack = packs()
    final = minimal_transport_state(pack.soc0, pack.capacity_ah, CONFIG.sim.soc_tolerance, 60)
    moved = (pack.capacity_ah * (pack.soc0 - final)).sum(axis=1)
    np.testing.assert_allclose(moved, 0.0, atol=1e-9)
    assert np.all(np.ptp(final, axis=1) <= CONFIG.sim.soc_tolerance + 1e-9)
    assert np.all(np.ptp(pack.soc0, axis=1) > CONFIG.sim.soc_tolerance)
    assert np.all(np.abs(final - pack.soc0).max(axis=1) > 0.0)


def test_minimal_transport_leaves_a_balanced_pack_alone() -> None:
    soc = np.array([[0.500, 0.503, 0.501, 0.504]])
    capacity = np.full((1, 4), 2.8)
    final = minimal_transport_state(soc, capacity, 0.01, 60)
    np.testing.assert_allclose(final, soc, atol=1e-9)


def test_energy_identity_matches_the_simulated_passive_dissipation() -> None:
    pack = packs()
    r_bleed = np.full(N_PACKS, 60.0)
    result = simulate(Technique.PASSIVE, pack, r_bleed, np.zeros(N_PACKS), CONFIG)
    identity = cell_energy_released_j(pack.soc0, result.final_soc, pack.capacity_ah, CONFIG.cell)
    np.testing.assert_allclose(identity, result.energy_dissipated_j, rtol=2e-3)


@pytest.mark.parametrize("technique", list(Technique))
def test_baselines_follow_the_simulator(technique: Technique) -> None:
    pack = packs()
    component, frequency = components(technique)
    simulated = simulate(technique, pack, component, frequency, CONFIG)
    predicted = physics_prediction(
        technique, pack.soc0, pack.capacity_ah, pack.r0_ohm, component, frequency, CONFIG
    )
    time_error = np.abs(predicted.balancing_time_s / simulated.balancing_time_s - 1.0)
    energy_error = np.abs(predicted.energy_dissipated_j / simulated.energy_dissipated_j - 1.0)
    limits = {
        Technique.PASSIVE: (0.03, 0.05),
        Technique.BUCK_BOOST: (0.10, 0.20),
        Technique.FLYING_CAPACITOR: (0.05, 0.03),
    }[technique]
    assert np.median(time_error) < limits[0]
    assert np.median(energy_error) < limits[1]
    assert np.all(predicted.balancing_time_s > 0.0)
    assert np.all(predicted.energy_dissipated_j > 0.0)


def test_flying_capacitor_solution_matches_a_matrix_exponential() -> None:
    pack = packs()
    component, frequency = components(Technique.FLYING_CAPACITOR)
    config = replace(CONFIG, ml_eval=replace(CONFIG.ml_eval, bisection_steps=60))
    prediction = flying_capacitor_prediction(
        pack.soc0, pack.capacity_ah, pack.r0_ohm, component, frequency, config
    )
    index = 3
    soc0, capacity = pack.soc0[index], pack.capacity_ah[index]
    mean = capacity_weighted_mean_soc(soc0[None], capacity[None])
    slope = float(ocv_slope_v_per_soc(mean, CONFIG.cell)[0])
    r_link = link_resistance_ohm(
        np.array([1.0 / (component[index] * frequency[index])]),
        pack.r0_ohm[index][None],
        CONFIG.flying_capacitor,
        frequency[index : index + 1],
    )[0]
    diff = np.zeros((3, 4))
    for k in range(3):
        diff[k, k], diff[k, k + 1] = 1.0, -1.0
    system = np.diag(r_link) + CONFIG.cell.r1_ohm * (diff @ diff.T)
    stiffness = slope / SECONDS_PER_HOUR * diff.T @ np.linalg.inv(system) @ diff
    state = expm(-np.diag(1.0 / capacity) @ stiffness * prediction.balancing_time_s[index]) @ soc0
    assert np.ptp(state) == pytest.approx(CONFIG.sim.soc_tolerance, abs=1e-6)


def test_passive_prediction_matches_the_closed_form_time() -> None:
    pack = packs()
    r_bleed = np.full(N_PACKS, 60.0)
    prediction = passive_prediction(pack.soc0, pack.capacity_ah, pack.r0_ohm, r_bleed, CONFIG)
    simulated = simulate(Technique.PASSIVE, pack, r_bleed, np.zeros(N_PACKS), CONFIG)
    error = np.abs(prediction.balancing_time_s / simulated.balancing_time_s - 1.0)
    assert error.max() < 0.03
