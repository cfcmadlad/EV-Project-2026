import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from src.cell_plots import (
    plot_cell_validation,
    plot_hg2_overview,
    plot_hppc_params,
    plot_validation_summary,
)
from src.config import CONFIG, MV_PER_V, PROJECT_ROOT, RESULTS_DIR
from src.experiments import random_streams
from src.hg2 import DRIVE_CYCLES, TestKind, load_all
from src.identification import (
    MODEL_2RC,
    candidate_models,
    check_models,
    constants,
    cross_check,
    fit_tables,
    identify,
    interval_table,
    propagate_inputs,
    replay_all,
    table_slope_mv_per_percent,
    usable_drive_cycles,
)
from src.identify import ocv_slope_v_per_soc
from src.params import ASSUMED

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hg2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARAMS_PATH = PROCESSED_DIR / "hg2_25c_params.json"
IDENTIFICATION_PATH = PROCESSED_DIR / "hg2_25c_identification.json"
BOOTSTRAP_PATH = PROCESSED_DIR / "hg2_25c_bootstrap.json"
PARAMETER_SET_NAME = "fitted_hg2_25c"
DATASET_TITLE = "LG 18650HG2 Li-ion Battery Data"
DATASET_DOI = "10.17632/b5mj79w5w9.2"
DATASET_LICENCE = "CC BY 4.0"
DATASET_INSTITUTION = "McMaster University"
ASSUMED_PARAMETERS = [
    "cell-to-cell capacity spread of +/-5 %",
    "cell-to-cell R0 spread of +/-10 %",
    "balancing hardware values (inductor, MOSFET, capacitor, bleed resistor ranges)",
]


def write_json(payload: object, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", newline="\n")


def main() -> int:
    if not RAW_DIR.exists():
        print(f"missing {RAW_DIR}; run scripts/fetch_hg2.py first")
        return 1
    start = perf_counter()
    spec = CONFIG.identification
    recordings = load_all(RAW_DIR)
    found = identify(recordings, spec)
    values = constants(found, spec)
    models = candidate_models(found, values)
    metrics, replays = replay_all(recordings, found, models, spec)
    selection = check_models(metrics, spec)
    kept = str(selection["kept_model"])
    simulator = models.lumped if kept == MODEL_2RC else models.one_rc
    rng = random_streams(CONFIG.seed)["identification"]
    samples = propagate_inputs(found, models, values, spec, rng)
    usable = usable_drive_cycles(recordings)
    unusable = tuple(kind.value for kind in DRIVE_CYCLES if not usable[kind])

    slope_samples = samples["ocv_slope_v_per_soc"] * 0.01 * MV_PER_V
    intervals = interval_table(
        {
            "r0_ohm": samples["r0_ohm"],
            "r1_ohm": samples["r1_ohm"],
            "c1_farad": samples["c1_farad"],
            "fast_r_ohm": samples["fast_r_ohm"],
            "fast_tau_s": samples["fast_tau_s"],
            "slow_r_ohm": samples["slow_r_ohm"],
            "slow_tau_s": samples["slow_tau_s"],
            "ocv_slope_mv_per_percent": slope_samples,
        },
        spec,
    )
    write_json(
        {
            "name": PARAMETER_SET_NAME,
            "temperature_c": 25,
            "source": {
                "title": DATASET_TITLE,
                "doi": DATASET_DOI,
                "licence": DATASET_LICENCE,
                "institution": DATASET_INSTITUTION,
            },
            "model": {
                "kept": kept,
                "simulator_branch": "two_rc_single_branch_equivalent"
                if kept == MODEL_2RC
                else "direct_one_rc",
                "validation_status": selection["status"],
            },
            "cell": {
                "capacity_ah": simulator.capacity_ah,
                "r0_ohm": simulator.r0_ohm,
                "r1_ohm": simulator.r1_ohm,
                "c1_farad": simulator.c1_farad,
                "ocv": {"soc": list(simulator.ocv.soc), "volt_v": list(simulator.ocv.volt_v)},
            },
            "constant_window_soc": list(spec.constant_window),
            "constant_window_summary": asdict(values),
            "assumed_parameters": ASSUMED_PARAMETERS,
            "soc_dependent_tables": {"enabled": False, **fit_tables(found)},
        },
        PARAMS_PATH,
    )
    write_json(
        {
            "n_resamples": spec.bootstrap_resamples,
            "block_samples": spec.bootstrap_block_samples,
            "samples": {name: array.tolist() for name, array in samples.items()},
        },
        BOOTSTRAP_PATH,
    )
    ocv = found.ocv
    write_json(
        {
            "capacity": asdict(found.capacity),
            "ocv": {
                "hysteresis_at_50_percent_mv": float(ocv.hysteresis_v[50] * MV_PER_V),
                "hysteresis_at_50_percent_literal_normalisation_mv": float(
                    ocv.hysteresis_literal_v[50] * MV_PER_V
                ),
                "slope_at_50_percent_mv_per_percent": ocv_slope_v_per_soc(ocv, 0.5)
                * 0.01
                * MV_PER_V,
                "slope_at_50_percent_assumed_mv_per_percent": table_slope_mv_per_percent(
                    ASSUMED.cell
                ),
                "first_to_final_pass_max_change_mv": float(
                    np.abs(ocv.ocv_v - found.ocv_first_pass.ocv_v).max() * MV_PER_V
                ),
                "monotone_adjustment_mv": ocv.monotone_adjustment_v * MV_PER_V,
                "endpoint_soc0_v": float(ocv.ocv_v[0]),
                "endpoint_soc100_v": float(ocv.ocv_v[-1]),
                "hppc_rest_cross_check": cross_check(found),
            },
            "hppc": fit_tables(found),
            "usable_drive_cycles": {kind.value: usable[kind] for kind in DRIVE_CYCLES},
            "validation": {
                cycle: {name: asdict(m) for name, m in per_model.items()}
                for cycle, per_model in metrics.items()
            },
            "selection": selection,
            "intervals": intervals,
            "runtime_note": "all values deterministic; timings are printed by the script only",
        },
        IDENTIFICATION_PATH,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    one_per_kind = {kind: items[0] for kind, items in recordings.items()}
    plot_hg2_overview(one_per_kind, RESULTS_DIR / "hg2_raw_overview", CONFIG.plot)
    plot_hppc_params(
        found.fits_1rc,
        found.fits_2rc,
        spec.constant_window,
        RESULTS_DIR / "hppc_params_vs_soc",
        CONFIG.plot,
    )
    plot_cell_validation(
        replays[TestKind.UDDS.value],
        kept,
        TestKind.UDDS.value,
        RESULTS_DIR / "cell_validation_udds",
        CONFIG.plot,
    )
    plot_validation_summary(
        metrics,
        unusable,
        kept,
        (spec.rmse_whole_cycle_v, spec.rmse_window_v),
        RESULTS_DIR / "cell_validation_summary",
        CONFIG.plot,
    )
    print(f"kept model {kept}: {selection['status']}; {perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
