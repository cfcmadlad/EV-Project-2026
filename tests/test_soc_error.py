from dataclasses import replace

import numpy as np

from src.balancing import simulate
from src.config import CONFIG, Technique
from src.pack import sample_packs
from src.params import SocErrorSpec
from src.soc_error import bias_draws, run_bias_study, sigma_label, soc_error_block

SPEC = SocErrorSpec(packs=12, max_balancing_h=6.0, deterministic_check_packs=4)
CONFIG_SMALL = replace(CONFIG, soc_error=SPEC)


def test_zero_sigma_reproduces_the_deterministic_simulation() -> None:
    block, _ = soc_error_block(CONFIG_SMALL, np.random.default_rng(1))
    assert block["sigma_zero_reproduces_deterministic"]


def test_zero_bias_is_bitwise_identical_to_no_bias() -> None:
    packs = sample_packs(5, np.random.default_rng(2), CONFIG.cell, CONFIG.pack)
    component = np.full(5, CONFIG.buck_boost.component.mid)
    frequency = np.full(5, CONFIG.buck_boost.frequency_hz.mid)
    plain = simulate(Technique.BUCK_BOOST, packs, component, frequency, CONFIG)
    zero = simulate(
        Technique.BUCK_BOOST, packs, component, frequency, CONFIG, soc_bias=np.zeros((5, 4))
    )
    assert np.array_equal(plain.final_soc, zero.final_soc)
    assert np.array_equal(plain.balancing_time_s, zero.balancing_time_s)


def test_bias_leaves_a_true_spread_for_passive_balancing() -> None:
    packs = sample_packs(10, np.random.default_rng(3), CONFIG.cell, CONFIG.pack)
    biases = bias_draws((0.0, 3.0), packs, np.random.default_rng(4))
    study = run_bias_study(packs, biases, CONFIG_SMALL)
    spread = study.final_spread[Technique.PASSIVE]
    assert np.median(spread[sigma_label(3.0)]) > np.median(spread[sigma_label(0.0)])
