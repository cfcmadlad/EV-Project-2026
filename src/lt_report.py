from dataclasses import replace
from pathlib import Path

import numpy as np

from src.balancing import buck_boost_currents, buck_boost_parameter
from src.calibration import LTSPICE_DIR, circuit_maps
from src.config import CONFIG
from src.ledger import buck_boost_element_loss_w
from src.table_io import read_rows

Rows = list[dict[str, str | float]]
NOT_RUN = {"status": "not_run"}
FILES = ("flying_cap.csv", "buck_boost.csv", "shunt.csv", "checks.csv")
CURRENT_TOLERANCE = 0.05
LOSS_TOLERANCE = 0.15
LOOSE_VARIANT = "loose_tol"
BASELINE_L_H = 55e-6
BASELINE_F_HZ = 30e3
BASELINE_C_F = 5.5e-3
BASELINE_FC_HZ = 5.5e3
BASELINE_SPREAD = 0.05


def _stat(errors: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(errors)),
        "mean": float(errors.mean()),
        "max_abs": float(np.abs(errors).max()),
        "n": int(errors.size),
    }


def _select(rows: Rows, **match: object) -> Rows:
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def flying_block(rows: Rows) -> dict[str, object]:
    out: dict[str, object] = {}
    for switch in ("real", "ideal"):
        chosen = _select(rows, switch=switch)
        lt = np.array([r["i_snk_a"] for r in chosen])
        delta_v = np.array([r["v_source"] - r["v_sink"] for r in chosen])
        series = np.array([r["py_i_series_a"] for r in chosen]) / lt - 1.0
        coth = np.array([r["py_i_coth_a"] for r in chosen]) / lt - 1.0
        loss_lt = np.array([r["p_loss_total_w"] - r["p_gate_w"] for r in chosen])
        loss_py = np.array([r["py_i_coth_a"] for r in chosen]) * delta_v
        loss_error = loss_py / loss_lt - 1.0
        out[switch] = {
            "current_error_series": _stat(series),
            "current_error_coth": _stat(coth),
            "loss_error_coth": _stat(loss_error),
            "best_form": "coth" if np.abs(coth).max() < np.abs(series).max() else "series",
        }
    real = _select(rows, switch="real")
    ideal = _select(rows, switch="ideal")
    grouped: dict[tuple, list[float]] = {}
    for r in real:
        grouped.setdefault((r["c_f"], r["f_hz"]), []).append(r["r_eff_ohm"])
    out["r_eff_spread_over_voltage"] = float(
        max((max(v) - min(v)) / np.mean(v) for v in grouped.values())
    )
    out["gate_power_w"] = {
        "min": min(r["p_gate_w"] for r in real),
        "max": max(r["p_gate_w"] for r in real),
    }
    out["efficiency_gap_pp"] = float(
        100.0 * max(abs(r["py_efficiency"] - r["efficiency_with_gate"]) for r in real)
    )
    r0_ratio = np.array(
        [
            (r["p_r0_src_w"] + r["p_r0_snk_w"])
            / (r["r0_ohm"] * (r["i_src_a"] ** 2 + r["i_snk_a"] ** 2))
            for r in real
        ]
    )
    out["r0_loss_vs_mean_current_form"] = _stat(r0_ratio - 1.0)
    out["real_vs_ideal_current"] = _stat(
        np.array([a["i_snk_a"] / b["i_snk_a"] - 1.0 for a, b in zip(real, ideal, strict=True)])
    )
    out["pss_change_max"] = float(max(r["pss_change_snk"] for r in rows))
    out["balance_residual_max"] = float(max(r["balance_residual_rel"] for r in rows))
    out["n_points"] = len(real)
    out["calibrated"] = flying_calibrated(real)
    return out


def flying_calibrated(real: Rows) -> dict[str, object]:
    maps = circuit_maps()
    errors = []
    for r in real:
        element = maps.flying_element_ohm(np.array(r["c_f"]), np.array(r["f_hz"]))
        current = (r["v_source"] - r["v_sink"]) / (element + 2.0 * r["r0_ohm"])
        errors.append(current / r["i_snk_a"] - 1.0)
    return {"current_error": _stat(np.array(errors))}


def calibrated_prediction(row: dict, circuit: bool) -> dict[str, float]:
    hardware = replace(CONFIG.buck_boost, circuit_calibrated=circuit)
    emf = np.array([[row["v_source"], row["v_sink"]]])
    r0 = np.full((1, 2), row["r0_ohm"])
    inverse = np.array([1.0 / (row["l_h"] * row["f_hz"])])
    config = replace(CONFIG, params=replace(CONFIG.params, buck_boost=hardware))
    parameter = buck_boost_parameter(
        config, inverse, np.array([row["l_h"]]), np.array([row["f_hz"]])
    )
    flow = buck_boost_currents(np.array([[1.0, 0.0]]), emf, emf, r0, parameter, hardware, 0.0)
    element = buck_boost_element_loss_w(flow, emf, r0, parameter, hardware)
    return {
        "i_src": float(flow.link_source_a[0, 0]),
        "i_snk": float(flow.link_sink_a[0, 0]),
        "loss": float(flow.removed_w[0] - flow.delivered_w[0]),
        "element_loss": float(element[0]),
        "efficiency": float(flow.delivered_w[0] / flow.removed_w[0]),
    }


def _bb_errors(chosen: Rows, circuit: bool) -> dict[str, dict[str, float]]:
    src, snk, loss, eff = [], [], [], []
    for r in chosen:
        p = calibrated_prediction(r, circuit)
        src.append(p["i_src"] / r["i_src_a"] - 1.0)
        snk.append(p["i_snk"] / r["i_snk_a"] - 1.0)
        measured = r["p_loss_total_w"] - (0.0 if circuit else r["p_gate_w"])
        loss.append(p["loss"] / measured - 1.0)
        eff.append(p["efficiency"] - (r["efficiency_with_gate"] if circuit else r["efficiency"]))
    return {
        "source_current": _stat(np.array(src)),
        "sink_current": _stat(np.array(snk)),
        "loss": _stat(np.array(loss)),
        "efficiency_abs": _stat(np.array(eff)),
    }


def buck_boost_block(rows: Rows) -> dict[str, object]:
    out: dict[str, object] = {}
    for switch in ("real", "ideal"):
        for rectifier in ("sync", "diode"):
            chosen = _select(rows, switch=switch, rectifier=rectifier)
            in_range = [r for r in chosen if r["saturates"] == 0.0]
            out[f"{switch}_{rectifier}"] = {
                "all_points": _bb_errors(chosen, False),
                "in_range": _bb_errors(in_range, False),
                "n_all": len(chosen),
                "n_in_range": len(in_range),
            }
    sync = _select(rows, switch="real", rectifier="sync")
    sync_in = [r for r in sync if r["saturates"] == 0.0]
    measured = np.array([r["p_loss_total_w"] - r["p_gate_w"] for r in sync_in])
    rms_form = np.array([r["py_loss_rms_w"] for r in sync_in]) / measured - 1.0
    mean_form = np.array([r["py_loss_mean_w"] for r in sync_in]) / measured - 1.0
    out["loss_error_mean_form"] = _stat(mean_form)
    out["loss_error_rms_form"] = _stat(rms_form)
    out["rms_form_closer"] = bool(np.median(np.abs(rms_form)) < np.median(np.abs(mean_form)))
    out["calibrated_in_range"] = _bb_errors(sync_in, True)
    out["calibrated_all_points"] = _bb_errors(sync, True)
    out["calibration_points"] = "mean over all three operating points per (L, f)"
    real = [r for r in rows if r["switch"] == "real"]
    out["duty_plus_d2_max"] = float(max(0.4 + r["d2_measured"] for r in real))
    out["idle_fraction_min"] = float(min(r["idle_fraction"] for r in real))
    extreme = [r for r in real if r["point"] == "v4p2_3p0"]
    out["extreme_duty_plus_d2_max"] = float(max(0.4 + r["d2_measured"] for r in extreme))
    out["saturating_points"] = int(sum(r["saturates"] for r in real))
    out["real_points"] = len(real)
    baseline = [r for r in sync if r["point"] == "soc55_50"]
    out["saturating_baseline_grid"] = sorted(
        [
            {
                "l_uh": r["l_h"] * 1e6,
                "f_khz": r["f_hz"] / 1e3,
                "i_pk_a": r["i_pk_a"],
                "isat_a": r["isat_a"],
            }
            for r in baseline
            if r["saturates"] == 1.0
        ],
        key=lambda d: (d["l_uh"], d["f_khz"]),
    )
    r0_ratio = np.array(
        [
            (r["p_r0_src_w"] + r["p_r0_snk_w"])
            / (r["r0_ohm"] * (r["i_src_a"] ** 2 + r["i_snk_a"] ** 2))
            for r in real
        ]
    )
    out["r0_loss_ratio_rms_over_mean_current"] = _stat(r0_ratio - 1.0)
    out["pss_change_max_real"] = float(max(r["pss_change_snk"] for r in real))
    out["pss_change_max_ideal"] = float(
        max(r["pss_change_snk"] for r in rows if r["switch"] == "ideal")
    )
    out["balance_residual_max"] = float(max(r["balance_residual_rel"] for r in rows))
    out["loss_breakdown_baseline"] = {
        "sync": loss_breakdown(_select(rows, switch="real", rectifier="sync", point="soc55_50")),
        "diode": loss_breakdown(_select(rows, switch="real", rectifier="diode", point="soc55_50")),
    }
    return out


BREAKDOWN_KEYS = (
    "p_r0_src_w",
    "p_r0_snk_w",
    "p_switch_cond_w",
    "p_switch_trans_w",
    "p_gate_w",
    "p_diode_w",
    "p_ind_w",
)


def loss_breakdown(rows: Rows) -> dict[str, float]:
    row = next(r for r in rows if r["l_h"] == BASELINE_L_H and r["f_hz"] == BASELINE_F_HZ)
    return {"l_uh": row["l_h"] * 1e6, "f_khz": row["f_hz"] / 1e3} | {
        key: row[key] for key in BREAKDOWN_KEYS
    }


def baseline_link(flying: Rows, buck: Rows) -> dict[str, object]:
    fc = next(
        r
        for r in flying
        if r["switch"] == "real"
        and r["c_f"] == BASELINE_C_F
        and r["f_hz"] == BASELINE_FC_HZ
        and abs(r["soc_spread"] - BASELINE_SPREAD) < 1e-9
    )
    element = circuit_maps().flying_element_ohm(np.array(fc["c_f"]), np.array(fc["f_hz"]))
    calibrated_current = (fc["v_source"] - fc["v_sink"]) / (element + 2.0 * fc["r0_ohm"])
    bb = next(
        r
        for r in buck
        if r["switch"] == "real"
        and r["rectifier"] == "sync"
        and r["point"] == "soc55_50"
        and r["l_h"] == BASELINE_L_H
        and r["f_hz"] == BASELINE_F_HZ
    )
    averaged, calibrated = calibrated_prediction(bb, False), calibrated_prediction(bb, True)
    return {
        "flying_capacitor": {
            "ltspice_current_a": fc["i_snk_a"],
            "averaged_current_a": fc["py_i_coth_a"],
            "calibrated_current_a": float(calibrated_current),
            "ltspice_efficiency": fc["efficiency_with_gate"],
            "averaged_efficiency": fc["py_efficiency"],
        },
        "buck_boost": {
            "ltspice_current_a": bb["i_snk_a"],
            "averaged_current_a": averaged["i_snk"],
            "calibrated_current_a": calibrated["i_snk"],
            "ltspice_efficiency": bb["efficiency_with_gate"],
            "averaged_efficiency": averaged["efficiency"],
            "calibrated_efficiency": calibrated["efficiency"],
        },
    }


def shunt_block(rows: Rows) -> dict[str, object]:
    return {
        "current_error": _stat(np.array([r["current_error"] for r in rows])),
        "resistor_power_error": _stat(
            np.array([r["p_resistor_w"] / r["py_p_resistor_w"] - 1.0 for r in rows])
        ),
        "n_points": len(rows),
    }


def checks_block(rows: Rows) -> dict[str, object]:
    by_name = {r["tag"]: r for r in rows}
    sensitivity: dict[str, float] = {}
    for r in rows:
        if r["variant"] == "base":
            continue
        base = by_name[r["tag"].replace(r["variant"], "base", 1)]
        change = max(
            abs(r[key] / base[key] - 1.0) for key in ("i_snk_a", "i_src_a", "p_loss_total_w")
        )
        sensitivity[r["variant"]] = max(sensitivity.get(r["variant"], 0.0), change)
    worst = max(v for k, v in sensitivity.items() if k != LOOSE_VARIANT)
    base_rows = [r for r in rows if r["variant"] == "base"]
    buck = [r for r in base_rows if r["converter"] == "buck_boost"]
    flying = [r for r in base_rows if r["converter"] == "flying_capacitor"]
    peak_error = []
    for r in buck:
        duty, period = 0.4, 1.0 / r["f_hz"]
        expected = (r["v_source"] - r["i_src_a"] * 0.0) * duty * period / r["component"]
        peak_error.append(r["i_pk_a"] / expected - 1.0)
    charge = [abs(r["cycle_charge_snk_c"] / r["cycle_charge_src_c"] - 1.0) for r in flying]
    return {
        "sensitivity_worst": worst,
        "sensitivity_loose": sensitivity[LOOSE_VARIANT],
        "sensitivity_by_variant": sensitivity,
        "peak_current_formula_error": _stat(np.array(peak_error)),
        "flying_cycle_charge_mismatch_max": float(max(charge)),
        "pss_change_max": float(max(r["pss_change_snk"] for r in rows)),
        "balance_residual_max": float(max(r["balance_residual_rel"] for r in base_rows)),
        "n_runs": len(rows),
    }


def default_decision(flying: dict, buck: dict) -> dict[str, object]:
    fc = flying["real"]
    bb = buck["real_sync"]["in_range"]
    checks = {
        "flying_capacitor_current": fc["current_error_coth"]["max_abs"] <= CURRENT_TOLERANCE,
        "flying_capacitor_loss": fc["loss_error_coth"]["max_abs"] <= LOSS_TOLERANCE,
        "buck_boost_current": max(bb["source_current"]["max_abs"], bb["sink_current"]["max_abs"])
        <= CURRENT_TOLERANCE,
        "buck_boost_loss": bb["loss"]["max_abs"] <= LOSS_TOLERANCE,
    }
    return {
        "averaged_within_tolerance": checks,
        "default_is_calibrated": not all(checks.values()),
    }


def ltspice_block(directory: Path = LTSPICE_DIR) -> dict[str, object]:
    if not all((directory / name).is_file() for name in FILES):
        return dict(NOT_RUN)
    flying = read_rows(directory / "flying_cap.csv")
    buck = read_rows(directory / "buck_boost.csv")
    shunt = read_rows(directory / "shunt.csv")
    checks = read_rows(directory / "checks.csv")
    flying_result = flying_block(flying)
    buck_result = buck_boost_block(buck)
    checks_result = checks_block(checks)
    return {
        "status": "ok",
        "energy_balance_residual_max": max(
            flying_result["balance_residual_max"],
            buck_result["balance_residual_max"],
            checks_result["balance_residual_max"],
        ),
        "default_decision": default_decision(flying_result, buck_result),
        "parts": {
            "flying_capacitor": "BSZ100N03LS",
            "buck_boost": "FDS4410",
            "schottky": "MBRS340",
        },
        "tolerances": {"current": CURRENT_TOLERANCE, "loss": LOSS_TOLERANCE},
        "flying_capacitor": flying_result,
        "buck_boost": buck_result,
        "baseline_link": baseline_link(flying, buck),
        "shunt": shunt_block(shunt),
        "checks": checks_result,
    }
