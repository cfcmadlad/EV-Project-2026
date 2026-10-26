import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import FloatArray, ProjectConfig

TEMPERATURES_C = (0, 10, 25, 40)
KELVIN_OFFSET = 273.15
BOLTZMANN_EV_PER_K = 8.617333262e-5


@dataclass(frozen=True, slots=True)
class ArrheniusFit:
    activation_ev: float
    ln_r0_at_reference: float
    reference_c: float

    def r0_ohm(self, temperature_c: FloatArray) -> FloatArray:
        inverse = 1.0 / (np.asarray(temperature_c) + KELVIN_OFFSET)
        reference = 1.0 / (self.reference_c + KELVIN_OFFSET)
        exponent = self.activation_ev / BOLTZMANN_EV_PER_K * (inverse - reference)
        return np.exp(self.ln_r0_at_reference + exponent)


def fit_arrhenius(
    temperature_c: FloatArray, r0_ohm: FloatArray, reference_c: float
) -> ArrheniusFit:
    inverse = 1.0 / (np.asarray(temperature_c) + KELVIN_OFFSET)
    reference = 1.0 / (reference_c + KELVIN_OFFSET)
    slope, intercept = np.polyfit(inverse - reference, np.log(r0_ohm), 1)
    return ArrheniusFit(
        activation_ev=float(slope * BOLTZMANN_EV_PER_K),
        ln_r0_at_reference=float(intercept),
        reference_c=reference_c,
    )


def bootstrap_activation_ev(
    temperature_c: FloatArray, r0_samples_ohm: FloatArray, reference_c: float
) -> FloatArray:
    return np.array(
        [
            fit_arrhenius(temperature_c, r0_samples_ohm[:, j], reference_c).activation_ev
            for j in range(r0_samples_ohm.shape[1])
        ]
    )


def linear_slope(temperature_c: FloatArray, values: FloatArray) -> float:
    return float(np.polyfit(temperature_c, values, 1)[0])


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CACHE_PATH = PROCESSED_DIR / "hg2_temperature.json"
PROVENANCE_PATH = PROCESSED_DIR / "provenance.json"
GATED_CYCLES = ("UDDS", "LA92")


def load_temperature_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    payload = json.loads(CACHE_PATH.read_text())
    hashes = {r["name"]: r["sha256"] for r in json.loads(PROVENANCE_PATH.read_text())["files"]}
    if any(hashes.get(name) != sha for name, sha in payload["inputs"].items()):
        return {"status": "stale_cache"}
    return payload


def temperature_block(config: ProjectConfig) -> dict[str, object]:
    payload = load_temperature_cache()
    if payload is None:
        return {"status": "not_run"}
    if payload.get("status") == "stale_cache":
        return {"status": "stale_cache"}
    spec = config.identification
    per_temperature: dict[str, object] = {}
    for key, entry in payload["per_temperature"].items():
        selection = entry["selection"]
        kept = selection["kept_model"]
        per_temperature[key] = {
            "capacity_ah": entry["capacity_ah"],
            "ocv_at_50_percent_v": entry["ocv_at_50_percent_v"],
            "ocv_slope_at_50_percent_mv_per_percent": entry[
                "ocv_slope_at_50_percent_mv_per_percent"
            ],
            "simulator_cell": entry["simulator_cell"],
            "kept_model": kept,
            "validation_status": selection["status"],
            "pass_by_model": selection["pass_by_model"],
            "gated_cycles": selection["gated_cycles"],
            "validation": {
                cycle: {
                    "rmse_whole_v": models[kept]["rmse_whole_v"],
                    "rmse_window_v": models[kept]["rmse_window_v"],
                    "passes": bool(
                        models[kept]["rmse_whole_v"] <= spec.rmse_whole_cycle_v
                        and models[kept]["rmse_window_v"] <= spec.rmse_window_v
                    ),
                }
                for cycle, models in entry["validation"].items()
                if cycle in GATED_CYCLES
            },
            "intervals": entry["intervals"],
        }
    return {
        "status": "ok",
        "temperatures_c": payload["temperatures_c"],
        "thresholds": {
            "rmse_whole_cycle_v": spec.rmse_whole_cycle_v,
            "rmse_window_v": spec.rmse_window_v,
        },
        "rejected_files": payload["rejected_files"],
        "per_temperature": per_temperature,
        "arrhenius": payload["arrhenius"],
        "ocv_shift": payload["ocv_shift"],
    }
