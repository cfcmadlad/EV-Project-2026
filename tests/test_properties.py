import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.balancing import build_control, simulate
from src.config import CONFIG, Technique
from src.diff_sim import build_twin, initial_state, tensor, twin_step
from src.ml_eval import calibration_quantile
from src.pack import sample_packs
from src.params import SECONDS_PER_HOUR
from src.stepper import PackState, cell_ocv, step

SLOW = settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)
SEEDS = st.integers(min_value=0, max_value=10_000)


def draw_packs(seed: int, n: int = 6):
    return sample_packs(n, np.random.default_rng(seed), CONFIG.cell, CONFIG.pack)


@given(
    seed=SEEDS,
    currents=st.lists(st.floats(min_value=-6.0, max_value=6.0), min_size=1, max_size=40),
)
@settings(max_examples=40, deadline=None, derandomize=True)
def test_soc_change_equals_integrated_current(seed, currents):
    packs = draw_packs(seed, 3)
    zeros = np.zeros(packs.n_packs)
    control = build_control(None, packs, zeros, zeros, CONFIG)
    state = PackState.initial(packs.soc0)
    for value in currents:
        state = step(state, np.full(packs.n_packs, value), control)
    drained_ah = (packs.soc0 - state.soc) * packs.capacity_ah
    expected = sum(currents) * CONFIG.sim.dt_s / SECONDS_PER_HOUR
    assert np.allclose(drained_ah, expected, rtol=1e-9, atol=1e-12)


@SLOW
@given(seed=SEEDS, technique=st.sampled_from([Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR]))
def test_ledger_closes_within_its_tolerance(seed, technique):
    packs = draw_packs(seed)
    spec = CONFIG.technique(technique)
    component = np.full(packs.n_packs, spec.component.mid)
    frequency = np.full(packs.n_packs, spec.frequency_hz.mid)
    result = simulate(technique, packs, component, frequency, CONFIG)
    gap = np.abs(result.ledger_j - result.energy_dissipated_j) / result.energy_removed_j
    assert gap.max() <= 0.01


@SLOW
@given(
    seed=SEEDS,
    resistances=st.tuples(
        st.floats(min_value=20.0, max_value=100.0), st.floats(min_value=20.0, max_value=100.0)
    ).filter(lambda pair: pair[1] > pair[0] * 1.05),
)
def test_passive_time_increases_with_bleed_resistance(seed, resistances):
    packs = draw_packs(seed)
    zeros = np.zeros(packs.n_packs)
    low = simulate(
        Technique.PASSIVE, packs, np.full(packs.n_packs, resistances[0]), zeros, CONFIG
    ).balancing_time_s
    high = simulate(
        Technique.PASSIVE, packs, np.full(packs.n_packs, resistances[1]), zeros, CONFIG
    ).balancing_time_s
    assert np.all(high > low)


@SLOW
@given(seed=SEEDS, technique=st.sampled_from(list(Technique)))
def test_same_seed_gives_identical_bytes(seed, technique):
    spec = CONFIG.technique(technique)
    runs = []
    for _ in range(2):
        packs = draw_packs(seed)
        component = np.full(packs.n_packs, spec.component.mid)
        frequency = np.full(packs.n_packs, spec.frequency_hz.mid)
        result = simulate(technique, packs, component, frequency, CONFIG)
        runs.append((result.balancing_time_s.tobytes(), result.energy_removed_j.tobytes()))
    assert runs[0] == runs[1]


@given(seed=SEEDS, n_calibration=st.integers(min_value=500, max_value=3000))
@settings(max_examples=25, deadline=None, derandomize=True)
def test_conformal_coverage_on_exchangeable_data(seed, n_calibration):
    rng = np.random.default_rng(seed)
    level = 0.9
    residual = np.abs(rng.standard_t(3, n_calibration))
    quantile = calibration_quantile(residual, level)
    fresh = np.abs(rng.standard_t(3, 40_000))
    coverage = float(np.mean(fresh <= quantile))
    assert abs(coverage - level) <= 5.0 / np.sqrt(n_calibration)


@given(seed=SEEDS, technique=st.sampled_from([Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR]))
@settings(max_examples=6, deadline=None, derandomize=True)
def test_torch_twin_matches_the_numpy_step(seed, technique):
    packs = draw_packs(seed, 8)
    spec = CONFIG.technique(technique)
    component = np.full(packs.n_packs, spec.component.mid)
    frequency = np.full(packs.n_packs, spec.frequency_hz.mid)
    control = build_control(
        technique, packs, component, frequency, CONFIG, track_load=True, gate_balancing=True
    )
    twin = build_twin(control, technique, CONFIG)
    reference = PackState.initial(packs.soc0, cell_ocv(packs.soc0, control))
    mirror = initial_state(twin, tensor(packs.soc0))
    rng = np.random.default_rng(seed)
    for _ in range(200):
        current = rng.uniform(-3.0, 3.0, packs.n_packs)
        reference = step(reference, current, control)
        mirror = twin_step(twin, mirror, tensor(current))
    assert np.allclose(mirror.soc.numpy(), reference.soc, rtol=1e-9, atol=1e-12)
    assert np.allclose(mirror.removed_j.numpy(), reference.removed_j, rtol=1e-9, atol=1e-9)
