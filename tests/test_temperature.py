import numpy as np
import pytest

from src.config import CONFIG
from src.temperature import (
    TEMPERATURES_C,
    ArrheniusFit,
    bootstrap_activation_ev,
    fit_arrhenius,
    load_temperature_cache,
    temperature_block,
)


def test_arrhenius_fit_recovers_known_parameters() -> None:
    truth = ArrheniusFit(0.25, float(np.log(0.012)), 25.0)
    temperature = np.array(TEMPERATURES_C, dtype=float)
    fit = fit_arrhenius(temperature, truth.r0_ohm(temperature), 25.0)
    assert fit.activation_ev == pytest.approx(0.25, rel=1e-9)
    assert np.exp(fit.ln_r0_at_reference) == pytest.approx(0.012, rel=1e-9)


def test_bootstrap_of_identical_samples_has_no_spread() -> None:
    temperature = np.array(TEMPERATURES_C, dtype=float)
    r0 = ArrheniusFit(0.2, float(np.log(0.01)), 25.0).r0_ohm(temperature)
    samples = np.repeat(r0[:, None], 5, axis=1)
    values = bootstrap_activation_ev(temperature, samples, 25.0)
    assert np.ptp(values) < 1e-12


def test_cached_temperature_fits_are_current() -> None:
    payload = load_temperature_cache()
    assert payload is not None and payload.get("status") != "stale_cache"
    block = temperature_block(CONFIG)
    assert block["status"] == "ok"
    assert block["temperatures_c"] == list(TEMPERATURES_C)
    for entry in block["per_temperature"].values():
        assert entry["capacity_ah"] > 2.0
