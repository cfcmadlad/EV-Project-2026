import numpy as np
import pytest

from src.headroom import (
    CELLS,
    CYCLES,
    HEADROOM_P90,
    NO_HEADROOM_MEDIAN,
    PACKS,
    gap,
    gap_stats,
    summarise,
    verdict,
)


def test_thresholds_are_fixed_in_advance():
    assert NO_HEADROOM_MEDIAN == 0.01
    assert HEADROOM_P90 == 0.02


def test_gap_is_relative_to_the_reference():
    assert gap(np.array([2.0]), np.array([1.9]))[0] == pytest.approx(0.05)
    assert gap(np.array([2.0]), np.array([2.0]))[0] == 0.0


def test_gap_stats_percentiles():
    stats = gap_stats(np.linspace(0.0, 1.0, 101))
    assert stats["median"] == pytest.approx(0.5)
    assert stats["p10"] == pytest.approx(0.1)
    assert stats["p90"] == pytest.approx(0.9)
    assert stats["max"] == 1.0 and stats["n"] == 101


@pytest.mark.parametrize(
    ("median", "p90", "none", "exists"),
    [(0.005, 0.015, True, False), (0.012, 0.03, False, True), (0.009, 0.05, True, True)],
)
def test_verdict_uses_the_preset_thresholds(median, p90, none, exists):
    result = verdict({"median": median, "p90": p90})
    assert result == {"no_meaningful_headroom": none, "headroom_exists": exists}


def test_summarise_pools_the_cycles():
    rng = np.random.default_rng(0)
    raw = {}
    for n in CELLS:
        for cycle in CYCLES:
            arrays = {"none": np.full(PACKS, 1.2), "reference": np.full(PACKS, 1.4)}
            for tech in ("passive", "buck_boost", "flying_capacitor"):
                for s in ("rule", "greedy", "rule_with_estimation_error"):
                    arrays[f"{tech}.{s}"] = 1.4 - rng.uniform(0.0, 0.02, PACKS)
            raw[(n, cycle)] = arrays
    result = summarise(raw)
    pooled = result["buck_boost"]["8"]["pooled"]["rule"]
    assert pooled["n"] == PACKS * len(CYCLES)
    assert 0.0 <= pooled["median"] <= 0.02 / 1.4
    assert set(result["passive"]) == {str(n) for n in CELLS}
