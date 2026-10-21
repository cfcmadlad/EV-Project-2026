import pytest

from src.config import CONFIG, Technique


@pytest.mark.parametrize("technique", list(Technique), ids=lambda t: t.key)
def test_ledger_closes_within_tolerance(physics: dict, technique: Technique) -> None:
    ledger = physics["energy_ledger"][technique.key]
    assert ledger["n_packs"] == CONFIG.checks.packs_per_technique
    assert ledger["max_net_vs_removed"] <= CONFIG.checks.ledger_tolerance


@pytest.mark.parametrize("technique", list(Technique), ids=lambda t: t.key)
def test_time_step_error_shrinks_with_dt(physics: dict, technique: Technique) -> None:
    scaling = physics["energy_ledger"][technique.key]["dt_scaling"]
    gaps = [
        scaling[f"dt_{dt_s:g}_s"]["max_per_link_vs_removed"]
        for dt_s in CONFIG.checks.dt_scaling_steps_s
    ]
    assert all(later < earlier for earlier, later in zip(gaps, gaps[1:], strict=False))
