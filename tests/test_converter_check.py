import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from src.config import CONFIG
from src.params import FlyingCapModel
from src.spice_check import (
    CACHE_PATH,
    DATASHEET_PATH,
    averaged_link,
    compare_case,
    converter_check,
    load_measurements,
    spice_cases,
    two_phase_closed_form_ohm,
)

RAW_DATASHEETS = Path(__file__).resolve().parents[1] / "data" / "raw" / "datasheets"
needs_cache = pytest.mark.skipif(not CACHE_PATH.exists(), reason="SPICE measurements not cached")


def case_named(name: str):
    return next(case for case in spice_cases(CONFIG) if case.name == name)


def test_buck_boost_average_matches_the_dcm_formula() -> None:
    case = case_named("buck_boost_s0_p1")
    link = averaged_link(case, CONFIG)
    conductance = 0.5 * 0.4**2 * (1.0 / (55e-6 * 30e3))
    expected = (
        case.cells.ocv_source_v * conductance / (1.0 + conductance * case.cells.r0_source_ohm)
    )
    assert link.source_current_a == pytest.approx(expected, rel=1e-12)
    assert link.sink_current_a < link.source_current_a
    assert link.total_loss_w > 0.0


def test_flying_capacitor_average_follows_the_closed_form() -> None:
    case = case_named("flying_capacitor_s0_p1")
    link = averaged_link(case, CONFIG, FlyingCapModel.TWO_PHASE)
    delta_v = case.cells.ocv_source_v - case.cells.ocv_sink_v
    assert link.sink_current_a == pytest.approx(
        delta_v / two_phase_closed_form_ohm(case, CONFIG), rel=1e-12
    )
    assert link.sink_current_a == pytest.approx(link.source_current_a, rel=1e-12)
    assert link.total_loss_w == pytest.approx(delta_v * link.sink_current_a, rel=1e-9)


def test_midsem_option_gives_the_larger_current_at_the_baseline_point() -> None:
    case = case_named("flying_capacitor_s0_p1")
    midsem = averaged_link(case, CONFIG, FlyingCapModel.MIDSEM)
    two_phase = averaged_link(case, CONFIG, FlyingCapModel.TWO_PHASE)
    assert 1.7 < midsem.sink_current_a / two_phase.sink_current_a < 1.9


def test_compare_case_applies_the_thresholds() -> None:
    case = case_named("flying_capacitor_s0_p1")
    link = averaged_link(case, CONFIG, FlyingCapModel.TWO_PHASE)
    cells = case.cells
    delta_v = cells.ocv_source_v - cells.ocv_sink_v
    exact = {
        "source_current_a": link.source_current_a,
        "sink_current_a": link.sink_current_a,
        "source_r0_loss_w": 0.4 * link.total_loss_w,
        "sink_r0_loss_w": 0.4 * link.total_loss_w,
        "switch_loss_direct_w": 0.2 * link.total_loss_w,
    }
    assert delta_v > 0.0
    good = compare_case(case, exact, link, CONFIG.converter_check)
    assert good["pass"]
    off = compare_case(
        case, {**exact, "sink_current_a": 0.9 * link.sink_current_a}, link, CONFIG.converter_check
    )
    assert not off["current_pass"]
    lossy = compare_case(
        case,
        {**exact, "switch_loss_direct_w": 5.0 * link.total_loss_w},
        link,
        CONFIG.converter_check,
    )
    assert not lossy["loss_pass"]


@needs_cache
def test_cached_measurements_match_the_current_configuration() -> None:
    payload = load_measurements(CONFIG)
    assert payload is not None
    assert payload["status"] == "ok"
    assert len(payload["cases"]) == 12
    assert all(len(record["netlist_sha256"]) == 64 for record in payload["cases"])


@needs_cache
def test_stale_cache_is_detected() -> None:
    changed = replace(
        CONFIG,
        params=replace(CONFIG.params, cell=replace(CONFIG.cell, r0_ohm=2.0 * CONFIG.cell.r0_ohm)),
    )
    assert load_measurements(changed) == {"status": "stale_cache"}
    block, rows = converter_check(changed)
    assert block["status"] == "stale_cache"
    assert rows == []


@needs_cache
def test_converter_check_verdicts() -> None:
    block, rows = converter_check(CONFIG)
    assert block["status"] == "ok"
    assert len(rows) == 18
    assert block["measurement_quality"]["pass"]
    assert block["buck_boost"]["pass"]
    assert block["flying_capacitor"]["two_phase"]["pass"]
    assert not block["flying_capacitor"]["midsem"]["pass"]
    assert block["flying_capacitor"]["midsem"]["n_pass"] == 2
    assert block["hypothesis"]["supported"]
    assert block["pass"]


@needs_cache
def test_quality_checks_meet_the_specified_limits() -> None:
    block, _ = converter_check(CONFIG)
    quality = block["measurement_quality"]
    spec = CONFIG.converter_check
    assert spec.max_step_divisor >= 400
    assert quality["stationarity"]["max_cycle_to_cycle_change"] < 1e-3
    assert quality["stationarity"]["measured_periods"] >= 50
    assert quality["buck_boost_dcm"]["min_idle_fraction"] > 0.0
    assert quality["flying_capacitor_phase_overlap"]["max_overlap_fraction"] == 0.0


@needs_cache
def test_netlist_hashes_match_the_files_in_spice_folder() -> None:
    import hashlib

    folder = Path(__file__).resolve().parents[1] / "spice"
    payload = json.loads(CACHE_PATH.read_text())
    for record in payload["cases"]:
        text = (folder / f"{record['name']}.cir").read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == record["netlist_sha256"]


def squeeze(text: str) -> str:
    return "".join(text.split())


def page_text(name: str, page: int) -> str:
    reader = PdfReader(RAW_DATASHEETS / name)
    return squeeze(reader.pages[page - 1].extract_text() or "")


def test_datasheet_values_file_is_complete() -> None:
    values = json.loads(DATASHEET_PATH.read_text())
    ti = values["bq76952"]["values"]
    assert ti["internal_balancing_resistance_ohm"] == {
        "min": 15.0,
        "typ": 28.0,
        "max": 46.0,
        "page": 18,
        "section": "7.20 Cell Balancing",
        "conditions": ti["internal_balancing_resistance_ohm"]["conditions"],
    }
    assert ti["maximum_balancing_current_per_cell_a"]["value"] == pytest.approx(0.1)
    adi = values["ltc3300_1"]["values"]
    assert adi["maximum_charge_transfer_efficiency_percent"]["value"] == 92.0
    assert adi["maximum_balancing_current_a"]["value"] == 10.0


@pytest.mark.skipif(not (RAW_DATASHEETS / "bq76952.pdf").exists(), reason="datasheet not present")
def test_bq76952_values_appear_on_the_cited_pages() -> None:
    values = json.loads(DATASHEET_PATH.read_text())["bq76952"]
    import hashlib

    assert (
        hashlib.sha256((RAW_DATASHEETS / "bq76952.pdf").read_bytes()).hexdigest()
        == values["sha256"]
    )
    resistance = values["values"]["internal_balancing_resistance_ohm"]
    text = page_text("bq76952.pdf", resistance["page"])
    assert squeeze("Internal cell balancing resistance") in text
    assert "152846" in text
    current = values["values"]["maximum_balancing_current_per_cell_a"]
    text = page_text("bq76952.pdf", current["page"])
    assert squeeze("Maximum cell balancing current") in text
    assert "100mA" in text


@pytest.mark.skipif(not (RAW_DATASHEETS / "LTC3300-1.pdf").exists(), reason="datasheet not present")
def test_ltc3300_values_appear_on_the_cited_page() -> None:
    values = json.loads(DATASHEET_PATH.read_text())["ltc3300_1"]
    import hashlib

    assert (
        hashlib.sha256((RAW_DATASHEETS / "LTC3300-1.pdf").read_bytes()).hexdigest()
        == values["sha256"]
    )
    text = page_text("LTC3300-1.pdf", 1)
    assert "UpTo92%ChargeTransferEfficiency".lower() in text.lower().replace(
        "t ransfer", "transfer"
    )
    assert "10ABalancingCurrent" in text


def test_ocv_at_half_soc_is_used_for_bleed_range() -> None:
    block, rows = converter_check(CONFIG)
    if block["status"] != "ok":
        pytest.skip("SPICE measurements not cached")
    sheet = block["datasheet_checks"]["bq76952"]
    low, high = sheet["model_bleed_current_a"]["range_at_50_percent_soc"]
    assert 0.03 < low < 0.04
    assert 0.18 < high < 0.19
    assert np.isclose(sheet["minimum_total_resistance_for_maximum_current_ohm"], 37.39, atol=0.05)
    assert sheet["internal_switch_only_exceeds_maximum"] == {"min": True, "typ": True, "max": False}
    assert len(rows) == 18
