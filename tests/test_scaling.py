from dataclasses import replace

import numpy as np
import pytest

from src.balancing import simulate
from src.config import CONFIG, Technique
from src.pack import PackBatch
from src.params import ScalingSpec
from src.scaling import (
    link_deadband,
    loglog_slope,
    path_laplacian_smallest,
    run_string,
    sample_string_packs,
    scaling_block,
    string_config,
    theory_slopes,
)


def test_deadband_is_unchanged_for_four_cells_and_shrinks_for_long_strings() -> None:
    assert link_deadband(CONFIG, 4) == CONFIG.sim.link_soc_deadband
    for n in (8, 16, 32):
        assert (n - 1) * link_deadband(CONFIG, n) < CONFIG.sim.soc_tolerance


def test_fixed_deadband_stalls_a_long_string_above_the_spread_target() -> None:
    n = 16
    soc0 = 0.5 + 0.0019 * (np.arange(n) - 7.5)[None, :]
    packs = PackBatch(
        capacity_ah=np.full((1, n), CONFIG.cell.capacity_ah),
        r0_ohm=np.full((1, n), CONFIG.cell.r0_ohm),
        soc0=soc0,
    )
    config = replace(CONFIG, sim=replace(CONFIG.sim, t_max_s=3600.0))
    result = simulate(Technique.BUCK_BOOST, packs, np.array([55e-6]), np.array([30e3]), config)
    assert not result.balanced[0]
    assert np.ptp(result.final_soc) > CONFIG.sim.soc_tolerance


def test_flying_capacitor_theory_slope_approaches_two() -> None:
    slopes = theory_slopes([8, 16, 32])
    assert slopes["flying_capacitor"] == pytest.approx(2.0, abs=0.05)
    assert path_laplacian_smallest(32) < path_laplacian_smallest(16)
    assert loglog_slope([2, 4, 8], [4.0, 16.0, 64.0]) == pytest.approx(2.0)


def test_physics_baselines_extend_to_eight_cells() -> None:
    config = replace(CONFIG, scaling=ScalingSpec(packs=6, max_balancing_h=100.0))
    packs = sample_string_packs(config, 8, np.random.default_rng(5))
    out = run_string(config, 8, packs, Technique.PASSIVE)
    assert out["balanced"].all()
    assert np.max(np.abs(out["physics_time_s"] / out["time_s"] - 1.0)) < 0.05
    assert string_config(config, 8).pack.n_cells == 8


def test_cached_scaling_block_is_current() -> None:
    block = scaling_block(CONFIG)
    assert block["status"] == "ok"
    assert block["cells"] == list(CONFIG.scaling.cells)
