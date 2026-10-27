import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from src.cell import build_ocv_curve
from src.cold_model import calibrate, replay_calibrated, table_model
from src.config import CONFIG, MV_PER_V, PROJECT_ROOT
from src.experiments import random_streams
from src.hg2 import TestKind, load_all
from src.identification import (
    MODEL_2RC,
    candidate_models,
    check_models,
    constants,
    fit_tables,
    identify,
    interval_table,
    propagate_inputs,
    replay_all,
    usable_drive_cycles,
)
from src.identify import interval, ocv_slope_v_per_soc
from src.params import CellParams, OCVTable
from src.soc_estimation.data import CYCLE_PATTERN, build_cycle
from src.temperature import (
    TEMPERATURES_C,
    bootstrap_activation_ev,
    fit_arrhenius,
    linear_slope,
)
from src.validate import replay_metrics

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hg2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "hg2_temperature.json"
PROVENANCE_PATH = PROCESSED_DIR / "provenance.json"
REFERENCE_C = 25
SLOPE_PER_PERCENT = 0.01
SOC_INDEX_50 = 50


def folder_for(temperature_c: int) -> Path:
    return RAW_DIR if temperature_c == REFERENCE_C else RAW_DIR / f"{temperature_c}degC"


def improved_model(folder, temperature: int, found, recordings, spec) -> dict[str, object]:
    calibration = CONFIG.calibration
    table = table_model(found.fits_2rc)
    curve = build_ocv_curve(models_cell(found))
    cycles = []
    used = []
    for path in sorted(folder.glob("*.csv")):
        match = CYCLE_PATTERN.match(path.name)
        if match is None or match["cycle"] not in calibration.calibration_cycles:
            continue
        try:
            cycles.append(
                build_cycle(
                    path,
                    match["cycle"],
                    temperature,
                    found.capacity_ah,
                    found.sign,
                    calibration.step_s,
                )
            )
        except ValueError:
            continue
        used.append(path.name)
    cycles = [c for c in cycles if c.n_steps > 1 and np.abs(c.current_a).max() > 1e-6]
    scales = calibrate(
        [np.arange(1, c.n_steps + 1) * calibration.step_s for c in cycles],
        [c.current_a for c in cycles],
        [c.voltage_v for c in cycles],
        [c.soc for c in cycles],
        table,
        curve,
    )
    validation = {}
    for kind in (TestKind.UDDS, TestKind.LA92):
        if kind not in recordings:
            continue
        replay = replay_calibrated(
            recordings[kind][0], found.sign, found.capacity_ah, table, curve, scales
        )
        metrics = replay_metrics(replay, spec)
        validation[kind.value] = {
            "rmse_whole_v": metrics.rmse_whole_v,
            "rmse_window_v": metrics.rmse_window_v,
        }
    return {
        "scales": scales.tolist(),
        "calibration_cycles": [c.name for c in cycles],
        "validation": validation,
    }


def models_cell(found):
    return CellParams(
        capacity_ah=found.capacity_ah,
        r0_ohm=0.01,
        r1_ohm=0.01,
        c1_farad=100.0,
        ocv=OCVTable(tuple(found.ocv.soc.tolist()), tuple(found.ocv.ocv_v.tolist())),
    )


def main() -> int:
    if not RAW_DIR.exists():
        print(f"missing {RAW_DIR}; run scripts/fetch_hg2.py --group temperature_fits first")
        return 1
    start = perf_counter()
    spec = CONFIG.identification
    hashes = {r["name"]: r["sha256"] for r in json.loads(PROVENANCE_PATH.read_text())["files"]}
    rngs = random_streams(CONFIG.seed)["identification"].spawn(len(TEMPERATURES_C))
    per_temperature: dict[str, object] = {}
    inputs: dict[str, str] = {}
    rejected: dict[str, str] = {}
    r0_samples: list[np.ndarray] = []
    r0_median: list[float] = []
    for temperature, rng in zip(TEMPERATURES_C, rngs, strict=True):
        rejected_here: dict[str, str] = {}
        recordings = load_all(folder_for(temperature), rejected=rejected_here)
        rejected |= rejected_here
        found = identify(recordings, spec)
        values = constants(found, spec)
        models = candidate_models(found, values)
        metrics, _ = replay_all(recordings, found, models, spec)
        selection = check_models(metrics, spec)
        kept = str(selection["kept_model"])
        simulator = models.lumped if kept == MODEL_2RC else models.one_rc
        samples = propagate_inputs(found, models, values, spec, rng)
        for kind_recordings in recordings.values():
            for recording in kind_recordings:
                inputs[recording.name] = hashes[recording.name]
        r0_samples.append(samples["r0_ohm"])
        r0_median.append(values.r0_ohm["median"])
        usable = usable_drive_cycles(recordings)
        per_temperature[str(temperature)] = {
            "capacity_ah": found.capacity_ah,
            "capacity_1c_ah": list(found.capacity.one_c_discharge_ah),
            "ocv": {"soc": found.ocv.soc.tolist(), "volt_v": found.ocv.ocv_v.tolist()},
            "hysteresis_half_width_v": found.ocv.hysteresis_v.tolist(),
            "ocv_at_50_percent_v": float(found.ocv.ocv_v[SOC_INDEX_50]),
            "ocv_slope_at_50_percent_mv_per_percent": ocv_slope_v_per_soc(found.ocv, 0.5)
            * SLOPE_PER_PERCENT
            * MV_PER_V,
            "constants": asdict(values),
            "simulator_cell": {
                "branch": "two_rc_single_branch_equivalent" if kept == MODEL_2RC else "one_rc",
                "r0_ohm": simulator.r0_ohm,
                "r1_ohm": simulator.r1_ohm,
                "c1_farad": simulator.c1_farad,
            },
            "one_rc_cell": {
                "r0_ohm": models.one_rc.r0_ohm,
                "r1_ohm": models.one_rc.r1_ohm,
                "c1_farad": models.one_rc.c1_farad,
            },
            "hppc_tables": fit_tables(found),
            "usable_drive_cycles": {k.value: v for k, v in usable.items()},
            "validation": {
                cycle: {
                    name: {
                        "rmse_whole_v": m.rmse_whole_v,
                        "rmse_window_v": m.rmse_window_v,
                        "soc_start": m.soc_start,
                        "soc_end": m.soc_end,
                    }
                    for name, m in per_model.items()
                }
                for cycle, per_model in metrics.items()
            },
            "selection": selection,
            "improved": improved_model(
                folder_for(temperature), temperature, found, recordings, spec
            ),
            "intervals": interval_table(
                {k: samples[k] for k in ("r0_ohm", "r1_ohm", "c1_farad")}, spec
            ),
        }
    temperature_c = np.array(TEMPERATURES_C, dtype=float)
    fit = fit_arrhenius(temperature_c, np.array(r0_median), REFERENCE_C)
    activation = bootstrap_activation_ev(temperature_c, np.stack(r0_samples), REFERENCE_C)
    ocv50 = np.array([per_temperature[str(t)]["ocv_at_50_percent_v"] for t in TEMPERATURES_C])
    shift_mv = (ocv50 - ocv50[TEMPERATURES_C.index(REFERENCE_C)]) * MV_PER_V
    predicted = fit.r0_ohm(temperature_c)
    payload = {
        "temperatures_c": list(TEMPERATURES_C),
        "reference_c": REFERENCE_C,
        "doi": json.loads(PROVENANCE_PATH.read_text())["dataset"]["doi"],
        "inputs": inputs,
        "rejected_files": rejected,
        "per_temperature": per_temperature,
        "arrhenius": {
            "form": "R0 = R0_ref exp(Ea/kB (1/T - 1/T_ref))",
            "activation_ev": fit.activation_ev,
            "r0_at_reference_ohm": float(np.exp(fit.ln_r0_at_reference)),
            "activation_ev_interval": interval(activation, spec.ci_percent),
            "n_resamples": int(activation.size),
            "r0_measured_ohm": r0_median,
            "r0_fitted_ohm": predicted.tolist(),
            "max_relative_residual": float(np.max(np.abs(predicted / np.array(r0_median) - 1.0))),
        },
        "ocv_shift": {
            "ocv_at_50_percent_v": ocv50.tolist(),
            "shift_vs_reference_mv": shift_mv.tolist(),
            "slope_mv_per_kelvin": linear_slope(temperature_c, ocv50) * MV_PER_V,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {OUTPUT_PATH.name} in {perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
