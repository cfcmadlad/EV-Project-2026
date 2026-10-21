from src.config import CONFIG


def test_flying_capacitor_conserves_charge(physics: dict) -> None:
    check = physics["flying_capacitor_charge"]
    assert check["max_relative_imbalance"] <= CONFIG.checks.charge_conservation_tolerance
