from dataclasses import replace

import numpy as np

from src.balancing import simulate
from src.checks import ledger_discrepancy, link_coupling_check, random_case
from src.config import CONFIG, Technique
from src.coupled_links import solve_tridiagonal
from src.pack import PackBatch
from src.params import LinkCoupling

NET = replace(CONFIG, sim=replace(CONFIG.sim, link_coupling=LinkCoupling.NET_CURRENT))


def two_cell_pack() -> PackBatch:
    return PackBatch(
        capacity_ah=np.full((3, 2), 2.8),
        r0_ohm=np.array([[0.010, 0.013], [0.012, 0.011], [0.009, 0.014]]),
        soc0=np.array([[0.56, 0.50], [0.53, 0.50], [0.60, 0.48]]),
    )


def test_single_link_has_no_shared_cell_and_both_models_agree() -> None:
    packs = two_cell_pack()
    n = packs.n_packs
    for technique, component, frequency in (
        (Technique.BUCK_BOOST, 55e-6, 30e3),
        (Technique.FLYING_CAPACITOR, 5e-3, 5e3),
    ):
        args = (technique, packs, np.full(n, component), np.full(n, frequency))
        per_link = simulate(*args, CONFIG)
        net = simulate(*args, NET)
        assert np.allclose(per_link.balancing_time_s, net.balancing_time_s, atol=1.0)
        assert np.allclose(per_link.final_soc, net.final_soc, atol=1e-6)


def test_tridiagonal_solver_matches_dense_solve() -> None:
    rng = np.random.default_rng(0)
    size = 6
    diagonal = rng.uniform(2.0, 3.0, (2, size))
    lower, upper = rng.uniform(-0.5, 0.0, (2, size)), rng.uniform(-0.5, 0.0, (2, size))
    lower[:, 0], upper[:, -1] = 0.0, 0.0
    rhs = rng.normal(size=(2, size))
    solved = solve_tridiagonal(lower, diagonal, upper, rhs)
    for i in range(2):
        matrix = np.diag(diagonal[i]) + np.diag(lower[i, 1:], -1) + np.diag(upper[i, :-1], 1)
        assert np.allclose(matrix @ solved[i], rhs[i])


def test_net_current_model_closes_the_net_ledger_for_both_active_techniques() -> None:
    rng = np.random.default_rng(1)
    for technique in (Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR):
        packs, component, frequency = random_case(technique, 20, CONFIG, rng)
        result = simulate(technique, packs, component, frequency, NET)
        assert result.balanced.all()
        assert ledger_discrepancy(result)["net_vs_dissipated"].max() < 0.01


def test_coupling_check_block_passes() -> None:
    small = replace(CONFIG, checks=replace(CONFIG.checks, coupling_packs=10))
    block = link_coupling_check(small, np.random.default_rng(2))
    assert block["pass"]
