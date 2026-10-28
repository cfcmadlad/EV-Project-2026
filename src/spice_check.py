import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.balancing import buck_boost_currents, flying_capacitor_currents
from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import PERCENT, FloatArray, ProjectConfig
from src.flow import StepFlow
from src.ledger import (
    buck_boost_element_loss_w,
    flying_capacitor_element_loss_w,
    per_link_r0_loss_w,
)
from src.params import ConverterCheckSpec, FlyingCapModel
from src.spice import LinkCells, RawTrace, SpiceCase
from src.switched_cap import link_resistances_ohm, two_phase_resistance_ohm

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CACHE_PATH = PROCESSED_DIR / "spice_measurements.json"
DATASHEET_PATH = PROCESSED_DIR / "datasheet_values.json"
NOT_RUN = {"status": "not_run"}
STALE = {"status": "stale_cache"}
PRESENT_FRACTION_FLOOR = 1e-3
DIRECTION_SOC = (1.0, 0.0)
BASELINE_FLYING_CASE = "flying_capacitor_s0_p1"
POINT_KEYS = (
    "case",
    "component",
    "frequency_hz",
    "spice_sink_a",
    "averaged_sink_a",
    "sink_error",
    "source_error",
    "spice_total_loss_w",
    "averaged_total_loss_w",
    "total_loss_error",
    "element_loss_error",
    "pass",
)


@dataclass(frozen=True, slots=True)
class AveragedLink:
    source_current_a: float
    sink_current_a: float
    removed_w: float
    delivered_w: float
    element_loss_w: float
    r0_loss_w: float
    sink_duty: float

    @property
    def total_loss_w(self) -> float:
        return self.removed_w - self.delivered_w


def spice_cases(config: ProjectConfig) -> list[SpiceCase]:
    spec = config.converter_check
    curve = build_ocv_curve(config.cell)
    r0 = config.cell.r0_ohm
    result: list[SpiceCase] = []
    for pair_index, (soc_a, soc_b) in enumerate(spec.soc_pairs):
        ocv = open_circuit_voltage(np.array([soc_a, soc_b]), curve)
        cells = LinkCells(float(ocv[0]), float(ocv[1]), r0, r0)
        for converter, points in (
            ("buck_boost", spec.buck_boost_points),
            ("flying_capacitor", spec.flying_capacitor_points),
        ):
            for point_index, (component, frequency) in enumerate(points):
                result.append(
                    SpiceCase(
                        name=f"{converter}_s{pair_index}_p{point_index}",
                        converter=converter,
                        cells=cells,
                        component=component,
                        frequency_hz=frequency,
                        step_divisor=spec.max_step_divisor,
                    )
                )
    return result


def case_inputs(config: ProjectConfig) -> dict[str, object]:
    spec = config.converter_check
    return {
        "soc_pairs": [list(pair) for pair in spec.soc_pairs],
        "cases": [
            {
                "name": case.name,
                "ocv_source_v": case.cells.ocv_source_v,
                "ocv_sink_v": case.cells.ocv_sink_v,
                "r0_ohm": case.cells.r0_source_ohm,
                "component": case.component,
                "frequency_hz": case.frequency_hz,
            }
            for case in spice_cases(config)
        ],
        "buck_boost": {
            "duty": config.buck_boost.duty,
            "r_inductor_ohm": config.buck_boost.r_inductor_ohm,
            "r_ds_on_ohm": config.buck_boost.r_ds_on_ohm,
        },
        "flying_capacitor": {
            "r_ds_on_ohm": config.flying_capacitor.r_ds_on_ohm,
            "esr_ohm": config.flying_capacitor.esr_ohm,
        },
        "periods": [spec.settle_periods, spec.measured_periods],
        "step_divisors": [spec.max_step_divisor, spec.refined_step_divisor],
    }


def _segment_fraction(
    trace: RawTrace, start_s: float, stop_s: float, values: FloatArray, threshold: float
) -> float:
    time_s = trace.time_s
    inside = (time_s[:-1] >= start_s) & (time_s[1:] <= stop_s)
    duration = np.diff(time_s)[inside]
    level = 0.5 * (np.abs(values[:-1]) + np.abs(values[1:]))[inside]
    return float(duration[level > threshold].sum() / duration.sum())


def waveform_diagnostics(
    case: SpiceCase, trace: RawTrace, start_s: float, spec: ConverterCheckSpec
) -> dict[str, float]:
    stop_s = start_s + spec.measured_periods * case.period_s
    window = (trace.time_s >= start_s) & (trace.time_s <= stop_s)
    if case.converter == "buck_boost":
        inductor = trace.signals["I(L1)"]
        peak = float(np.abs(inductor[window]).max())
        floor = PRESENT_FRACTION_FLOOR * peak
        return {
            "peak_inductor_current_a": peak,
            "switch_on_fraction": _segment_fraction(
                trace, start_s, stop_s, trace.signals["I(S1)"], floor
            ),
            "freewheel_fraction": _segment_fraction(
                trace, start_s, stop_s, trace.signals["I(D1)"], floor
            ),
            "idle_fraction": 1.0 - _segment_fraction(trace, start_s, stop_s, inductor, floor),
        }
    source = trace.signals["I(VA)"]
    peak = float(np.abs(source[window]).max())
    floor = PRESENT_FRACTION_FLOOR * peak
    both = np.minimum(np.abs(trace.signals["I(S1A)"]), np.abs(trace.signals["I(S2A)"]))
    return {
        "peak_source_current_a": peak,
        "phase_overlap_fraction": _segment_fraction(trace, start_s, stop_s, both, floor),
    }


def averaged_link(
    case: SpiceCase, config: ProjectConfig, model: FlyingCapModel = FlyingCapModel.TWO_PHASE
) -> AveragedLink:
    cells = case.cells
    soc = np.array([DIRECTION_SOC])
    ocv = np.array([[cells.ocv_source_v, cells.ocv_sink_v]])
    r0 = np.array([[cells.r0_source_ohm, cells.r0_sink_ohm]])
    parameter = np.array([1.0 / (case.component * case.frequency_hz)])
    deadband = config.sim.link_soc_deadband
    if case.converter == "buck_boost":
        flow = buck_boost_currents(soc, ocv, ocv, r0, parameter, config.buck_boost, deadband)
        element = buck_boost_element_loss_w(flow, ocv, r0, parameter, config.buck_boost)
        sink_duty = _buck_boost_sink_duty(flow, ocv, r0, config)
    else:
        hardware = replace(config.flying_capacitor, model=model)
        resistance = link_resistances_ohm(parameter, r0, hardware)
        flow = flying_capacitor_currents(soc, ocv, ocv, r0, resistance, deadband)
        element = flying_capacitor_element_loss_w(flow, ocv, r0, resistance)
        sink_duty = 0.5
    return _link_summary(flow, element, r0, sink_duty)


def _buck_boost_sink_duty(
    flow: StepFlow, ocv: FloatArray, r0: FloatArray, config: ProjectConfig
) -> float:
    terminal = ocv[0, 0] - flow.link_source_a[0, 0] * r0[0, 0]
    return float(config.buck_boost.duty * terminal / ocv[0, 1])


def _link_summary(
    flow: StepFlow, element: FloatArray, r0: FloatArray, sink_duty: float
) -> AveragedLink:
    return AveragedLink(
        source_current_a=float(flow.link_source_a[0, 0]),
        sink_current_a=float(flow.link_sink_a[0, 0]),
        removed_w=float(flow.removed_w[0]),
        delivered_w=float(flow.delivered_w[0]),
        element_loss_w=float(element[0]),
        r0_loss_w=float(per_link_r0_loss_w(flow, r0)[0]),
        sink_duty=sink_duty,
    )


def relative_error(averaged: float, reference: float) -> float:
    return averaged / reference - 1.0


def compare_case(
    case: SpiceCase, record: dict[str, object], averaged: AveragedLink, spec: ConverterCheckSpec
) -> dict[str, object]:
    cells = case.cells
    spice_source = float(record["source_current_a"])
    spice_sink = float(record["sink_current_a"])
    spice_removed = cells.ocv_source_v * spice_source
    spice_delivered = cells.ocv_sink_v * spice_sink
    spice_total = spice_removed - spice_delivered
    spice_r0 = float(record["source_r0_loss_w"]) + float(record["sink_r0_loss_w"])
    spice_element = float(record["switch_loss_direct_w"])
    spice_total = spice_r0 + spice_element
    row: dict[str, object] = {
        "case": case.name,
        "converter": case.converter,
        "component": case.component,
        "frequency_hz": case.frequency_hz,
        "ocv_difference_v": cells.ocv_source_v - cells.ocv_sink_v,
        "spice_source_a": spice_source,
        "spice_sink_a": spice_sink,
        "averaged_source_a": averaged.source_current_a,
        "averaged_sink_a": averaged.sink_current_a,
        "source_error": relative_error(averaged.source_current_a, spice_source),
        "sink_error": relative_error(averaged.sink_current_a, spice_sink),
        "spice_total_loss_w": spice_total,
        "averaged_total_loss_w": averaged.total_loss_w,
        "total_loss_error": relative_error(averaged.total_loss_w, spice_total),
        "spice_element_loss_w": spice_element,
        "averaged_element_loss_w": averaged.element_loss_w,
        "element_loss_error": relative_error(averaged.element_loss_w, spice_element),
        "spice_efficiency": spice_delivered / spice_removed,
        "averaged_efficiency": averaged.delivered_w / averaged.removed_w,
        "spice_power_difference_w": spice_removed - spice_delivered,
        "spice_energy_balance_gap": abs(spice_removed - spice_delivered - spice_total)
        / spice_removed,
    }
    row["current_pass"] = bool(
        max(abs(row["source_error"]), abs(row["sink_error"])) <= spec.current_tolerance
    )
    row["loss_pass"] = bool(abs(row["total_loss_error"]) <= spec.loss_tolerance)
    row["pass"] = bool(row["current_pass"] and row["loss_pass"])
    return row


def two_phase_closed_form_ohm(case: SpiceCase, config: ProjectConfig) -> float:
    hardware = config.flying_capacitor
    r_phase = case.cells.r0_source_ohm + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    r_other = case.cells.r0_sink_ohm + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
    parameter = np.array([1.0 / (case.component * case.frequency_hz)])
    return float(two_phase_resistance_ohm(parameter, np.array([r_phase]), np.array([r_other]))[0])


def series_link_resistance_ohm(case: SpiceCase, config: ProjectConfig) -> float:
    hardware = config.flying_capacitor
    return (
        1.0 / (case.component * case.frequency_hz)
        + 2.0 * hardware.r_ds_on_ohm
        + hardware.esr_ohm
        + case.cells.r0_source_ohm
        + case.cells.r0_sink_ohm
    )


def load_measurements(config: ProjectConfig) -> dict | None:
    if not CACHE_PATH.exists():
        return None
    payload = json.loads(CACHE_PATH.read_text())
    expected = json.loads(json.dumps(case_inputs(config)))
    if payload.get("inputs") != expected or payload.get("parameter_set") != config.params.name:
        return {"status": "stale_cache"}
    return payload


def _models_for(case: SpiceCase) -> tuple[tuple[str, FlyingCapModel], ...]:
    if case.converter == "buck_boost":
        return (("averaged", FlyingCapModel.TWO_PHASE),)
    return (("series", FlyingCapModel.SERIES), ("two_phase", FlyingCapModel.TWO_PHASE))


def comparison_rows(config: ProjectConfig, payload: dict) -> list[dict[str, object]]:
    cases = {case.name: case for case in spice_cases(config)}
    rows: list[dict[str, object]] = []
    for record in payload["cases"]:
        case = cases[record["name"]]
        for label, model in _models_for(case):
            averaged = averaged_link(case, config, model)
            row = compare_case(case, record, averaged, config.converter_check)
            rows.append({**row, "model": label})
    return rows


def _summarise_model(
    rows: list[dict[str, object]], converter: str, model: str
) -> dict[str, object]:
    chosen = [row for row in rows if row["converter"] == converter and row["model"] == model]
    return {
        "n_points": len(chosen),
        "n_pass": sum(bool(row["pass"]) for row in chosen),
        "max_abs_current_error": max(
            max(abs(row["source_error"]), abs(row["sink_error"])) for row in chosen
        ),
        "max_abs_total_loss_error": max(abs(row["total_loss_error"]) for row in chosen),
        "max_abs_element_loss_error": max(abs(row["element_loss_error"]) for row in chosen),
        "pass": all(bool(row["pass"]) for row in chosen),
        "points": [{key: row[key] for key in POINT_KEYS} for row in chosen],
    }


def measurement_quality(
    payload: dict, rows: list[dict[str, object]], spec: ConverterCheckSpec
) -> dict[str, object]:
    records = payload["cases"]
    cycle = max(max(r["cycle_change_source"], r["cycle_change_sink"]) for r in records)
    refinements = [r["step_refinement_change"] for r in records if "step_refinement_change" in r]
    balance = max(row["spice_energy_balance_gap"] for row in rows)
    bucks = [r for r in records if r["name"].startswith("buck_boost")]
    flyings = [r for r in records if r["name"].startswith("flying_capacitor")]
    idle = min(r["idle_fraction"] for r in bucks)
    overlap = max(r["phase_overlap_fraction"] for r in flyings)
    checks = {
        "stationarity": {
            "max_cycle_to_cycle_change": cycle,
            "threshold": spec.stationarity_tolerance,
            "measured_periods": spec.measured_periods,
            "pass": bool(cycle <= spec.stationarity_tolerance and spec.measured_periods >= 50),
        },
        "step_independence": {
            "max_current_change_at_half_step": max(refinements),
            "threshold": spec.step_independence_tolerance,
            "pass": bool(max(refinements) <= spec.step_independence_tolerance),
        },
        "energy_balance": {
            "max_gap_fraction_of_removed_power": balance,
            "threshold": spec.energy_balance_tolerance,
            "pass": bool(balance <= spec.energy_balance_tolerance),
        },
        "buck_boost_dcm": {
            "min_idle_fraction": idle,
            "threshold": spec.dcm_idle_fraction_min,
            "pass": bool(idle >= spec.dcm_idle_fraction_min),
        },
        "flying_capacitor_phase_overlap": {
            "max_overlap_fraction": overlap,
            "pass": bool(overlap == 0.0),
        },
    }
    return {**checks, "pass": all(bool(item["pass"]) for item in checks.values())}


def hypothesis_block(
    config: ProjectConfig, payload: dict, rows: list[dict[str, object]]
) -> dict[str, object]:
    cases = {case.name: case for case in spice_cases(config)}
    hardware = config.flying_capacitor
    points = []
    for record in payload["cases"]:
        case = cases[record["name"]]
        if case.converter != "flying_capacitor":
            continue
        by_model = {
            row["model"]: row
            for row in rows
            if row["case"] == case.name and row["converter"] == "flying_capacitor"
        }
        r_phase = case.cells.r0_source_ohm + 2.0 * hardware.r_ds_on_ohm + hardware.esr_ohm
        delta_v = case.cells.ocv_source_v - case.cells.ocv_sink_v
        spice_ohm = delta_v / float(record["sink_current_a"])
        series_ohm = series_link_resistance_ohm(case, config)
        closed_ohm = two_phase_closed_form_ohm(case, config)
        points.append(
            {
                "case": case.name,
                "component": case.component,
                "frequency_hz": case.frequency_hz,
                "phase_resistance_ohm": r_phase,
                "phase_time_constant_s": r_phase * case.component,
                "half_period_s": 0.5 * case.period_s,
                "regime": "fast" if r_phase * case.component > 0.5 * case.period_s else "slow",
                "slow_limit_ohm": 1.0 / (case.component * case.frequency_hz),
                "fast_limit_ohm": 4.0 * r_phase,
                "spice_ohm": spice_ohm,
                "two_phase_closed_form_ohm": closed_ohm,
                "series_ohm": series_ohm,
                "spice_over_series": spice_ohm / series_ohm,
                "spice_over_closed_form": spice_ohm / closed_ohm,
                "series_current_error": by_model["series"]["sink_error"],
                "two_phase_current_error": by_model["two_phase"]["sink_error"],
            }
        )
    baseline = next(p for p in points if p["case"] == BASELINE_FLYING_CASE)
    tolerance = config.converter_check.current_tolerance
    return {
        "points": points,
        "baseline_point": baseline,
        "baseline_series_current_overstatement_factor": 1.0 + baseline["series_current_error"],
        "supported": bool(
            abs(baseline["series_current_error"]) > tolerance
            and abs(baseline["two_phase_current_error"]) <= tolerance
            and baseline["regime"] == "fast"
        ),
    }


def datasheet_checks(config: ProjectConfig, rows: list[dict[str, object]]) -> dict[str, object]:
    if not DATASHEET_PATH.exists():
        return dict(NOT_RUN)
    sheets = json.loads(DATASHEET_PATH.read_text())
    curve = build_ocv_curve(config.cell)
    ocv_half, ocv_full = (float(v) for v in open_circuit_voltage(np.array([0.5, 1.0]), curve))
    r0 = config.cell.r0_ohm
    bleed = config.passive.component
    ti = sheets["bq76952"]["values"]
    internal = ti["internal_balancing_resistance_ohm"]
    limit = ti["maximum_balancing_current_per_cell_a"]
    min_total_ohm = ocv_half / limit["value"] - r0
    internal_a = {key: ocv_half / internal[key] for key in ("min", "typ", "max")}
    adi = sheets["ltc3300_1"]["values"]
    efficiency = adi["maximum_charge_transfer_efficiency_percent"]
    buck_rows = [
        row
        for row in rows
        if row["converter"] == "buck_boost" and str(row["case"]).startswith("buck_boost_s0_")
    ]
    operating_points = [
        {
            "case": row["case"],
            "spice_efficiency_percent": row["spice_efficiency"] * PERCENT,
            "averaged_efficiency_percent": row["averaged_efficiency"] * PERCENT,
            "spice_mean_current_a": row["spice_sink_a"],
        }
        for row in buck_rows
    ]
    within = (bleed.high - max(min_total_ohm, bleed.low)) / (bleed.high - bleed.low)
    return {
        "bq76952": {
            "document_revision": sheets["bq76952"]["revision"],
            "internal_resistance_ohm": {key: internal[key] for key in ("min", "typ", "max")},
            "internal_resistance_page": internal["page"],
            "maximum_current_a": limit["value"],
            "maximum_current_page": limit["page"],
            "model_bleed_resistance_range_ohm": [bleed.low, bleed.high],
            "model_bleed_current_a": {
                "range_at_50_percent_soc": [
                    ocv_half / (bleed.high + r0),
                    ocv_half / (bleed.low + r0),
                ],
                "high_end_at_100_percent_soc": ocv_full / (bleed.low + r0),
            },
            "internal_switch_only_current_at_50_percent_soc_a": internal_a,
            "internal_switch_only_exceeds_maximum": {
                key: bool(value > limit["value"]) for key, value in internal_a.items()
            },
            "minimum_total_resistance_for_maximum_current_ohm": min_total_ohm,
            "fraction_of_model_bleed_range_within_maximum": float(np.clip(within, 0.0, 1.0)),
        },
        "ltc3300_1": {
            "document_revision": sheets["ltc3300_1"]["revision"],
            "vendor_best_efficiency_percent": efficiency["value"],
            "vendor_best_efficiency_page": efficiency["page"],
            "vendor_maximum_current_a": adi["maximum_balancing_current_a"]["value"],
            "model_operating_points": operating_points,
            "model_exceeds_vendor_best": [
                bool(point["spice_efficiency_percent"] > efficiency["value"])
                for point in operating_points
            ],
            "model_efficiency_is_upper_bound": True,
            "omitted_losses": "switching transition, gate drive and core losses",
            "topology_note": "transformer-based bidirectional flyback; range check only",
        },
    }


def converter_check(
    config: ProjectConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = load_measurements(config)
    if payload is None:
        return dict(NOT_RUN), []
    if payload.get("status") == "stale_cache":
        return dict(STALE), []
    spec = config.converter_check
    rows = comparison_rows(config, payload)
    quality = measurement_quality(payload, rows, spec)
    buck = _summarise_model(rows, "buck_boost", "averaged")
    series_form = _summarise_model(rows, "flying_capacitor", "series")
    two_phase = _summarise_model(rows, "flying_capacitor", "two_phase")
    active = series_form if config.flying_capacitor.model is FlyingCapModel.SERIES else two_phase
    block: dict[str, object] = {
        "status": "ok",
        "tool": payload["tool"],
        "parameter_set": payload["parameter_set"],
        "thresholds": {
            "mean_current": spec.current_tolerance,
            "total_loss": spec.loss_tolerance,
        },
        "measurement_quality": quality,
        "buck_boost": buck,
        "flying_capacitor": {"series": series_form, "two_phase": two_phase},
        "flying_capacitor_active_model": config.flying_capacitor.model.value,
        "hypothesis": hypothesis_block(config, payload, rows),
        "datasheet_checks": datasheet_checks(config, rows),
        "pass": bool(quality["pass"] and buck["pass"] and active["pass"]),
    }
    return block, rows
