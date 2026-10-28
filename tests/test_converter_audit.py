from dataclasses import replace

import numpy as np
import pytest

from src.balancing import simulate
from src.checks import ledger_discrepancy, random_case
from src.config import CONFIG, Technique
from src.converter_audit import (
    flying_capacitor_audit,
    peak_current_audit,
    ripple_audit,
    with_buck_boost,
)
from src.switched_cap import coth_resistance_ohm, two_phase_resistance_ohm


def test_peak_current_worst_corner_and_dcm() -> None:
    audit = peak_current_audit(CONFIG)
    assert audit["worst_case"]["peak_current_a"] == pytest.approx(4.2 * 0.4 / (10e-6 * 10e3))
    assert audit["dcm"]["discontinuous_everywhere"]
    assert audit["dcm"]["max_d_plus_d2"] == pytest.approx(0.4 * (1.0 + 1.4))


def test_coth_form_equals_two_phase_for_equal_phases() -> None:
    inverse = np.geomspace(1e-3, 1.0, 20)
    r_phase = np.full(20, 0.04)
    assert np.allclose(
        coth_resistance_ohm(inverse, r_phase), two_phase_resistance_ohm(inverse, r_phase, r_phase)
    )


def test_flying_capacitor_audit_reports_the_circuit_default_and_orders_the_forms() -> None:
    audit = flying_capacitor_audit(CONFIG)
    assert audit["default_model"] == "circuit"
    runs = audit["baseline"]
    assert runs["coth"]["balancing_time_s"] > runs["midsem"]["balancing_time_s"]
    assert runs["coth"]["balancing_time_s"] == pytest.approx(
        runs["two_phase"]["balancing_time_s"], rel=1e-3
    )


def test_rms_current_flag_raises_energy_and_keeps_the_ledger_closed() -> None:
    small = replace(CONFIG, checks=replace(CONFIG.checks, audit_packs=8))
    audit = ripple_audit(small, np.random.default_rng(4))
    assert audit["dataset_range"]["energy_ratio_median"] > 1.0
    assert audit["dataset_range"]["ledger_gap_vs_removed_max_rms_model"] < 0.01
    assert not audit["default_uses_rms_current"]
    flagged = with_buck_boost(CONFIG, ripple_r0_loss=True)
    packs, component, frequency = random_case(
        Technique.BUCK_BOOST, 6, CONFIG, np.random.default_rng(2)
    )
    result = simulate(Technique.BUCK_BOOST, packs, component, frequency, flagged)
    assert result.balanced.all()
    assert ledger_discrepancy(result)["net_vs_removed"].max() < 0.01
