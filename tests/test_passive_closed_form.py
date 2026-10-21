from src.config import CONFIG


def test_passive_closed_form_matches_simulator(physics: dict) -> None:
    check = physics["passive_closed_form"]
    tolerance = CONFIG.checks.passive_closed_form_tolerance
    assert check["baseline_relative_error"] <= tolerance
    assert check["max_relative_error"] <= tolerance
    assert check["pass"]
