import numpy as np
import pytest

from src.config import CONFIG, Technique
from src.drive_cycle import (
    PROFILES_PATH,
    Mode,
    constant_program,
    ledger_gap,
    load_profiles,
    program_from_profiles,
    run_scenario,
    soc_conservation_error,
)
from src.experiments import baseline_pack
from src.pack import tile_pack

needs_profiles = pytest.mark.skipif(not PROFILES_PATH.exists(), reason="profiles not prepared")
PACK = baseline_pack(CONFIG)


def test_charge_stops_at_the_upper_voltage_limit() -> None:
    program = constant_program(-CONFIG.cell.capacity_ah * 0.5, 1, 10800)
    run = run_scenario(None, PACK, program, Mode.CHARGE, CONFIG)
    assert run.limit_reached.all()
    assert run.usable_ah[0] > 0.5


def test_soc_change_matches_coulomb_counting_with_balancing_off() -> None:
    profile = np.tile(np.array([5.0, -2.0, 8.0, 0.0]), 50)
    program = program_from_profiles([profile])
    run = run_scenario(None, PACK, program, Mode.DISCHARGE, CONFIG)
    assert soc_conservation_error(run, PACK.capacity_ah) <= 1e-9


def test_ledger_closes_under_constant_load_for_every_technique() -> None:
    program = constant_program(3.0, 1, 1200)
    for technique in Technique:
        run = run_scenario(technique, PACK, program, Mode.DISCHARGE, CONFIG)
        assert ledger_gap(run)["max_vs_removed"] <= 0.01


@needs_profiles
def test_cached_profiles_match_provenance_and_conserve_charge() -> None:
    payload = load_profiles(CONFIG.drive_cycle)
    assert payload is not None and payload.get("status") != "stale_cache"
    profile = payload["profiles"]["Mixed1"]
    ah = sum(profile["current_a"]) / 3600.0
    assert ah == pytest.approx(profile["discharge_ah"], rel=1e-12)
    assert tile_pack(PACK, 2).n_packs == 2
