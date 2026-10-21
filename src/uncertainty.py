import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.balancing import simulate
from src.config import FloatArray, ProjectConfig, Technique
from src.pack import PackBatch, tile_pack

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
IDENTIFICATION_PATH = PROCESSED_DIR / "hg2_25c_identification.json"
BOOTSTRAP_PATH = PROCESSED_DIR / "hg2_25c_bootstrap.json"
NOT_RUN = {"status": "not_run"}


def load_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resampled_packs(
    pack: PackBatch, samples: dict[str, list[float]], config: ProjectConfig
) -> PackBatch:
    n = config.identification.bootstrap_resamples
    r0_scale = np.array(samples["r0_ohm"][:n]) / config.cell.r0_ohm
    tiled = tile_pack(pack, r0_scale.size)
    return replace(
        tiled,
        r0_ohm=tiled.r0_ohm * r0_scale[:, None],
        r1_ohm=np.array(samples["r1_ohm"][:n]),
        c1_farad=np.array(samples["c1_farad"][:n]),
        ocv_tilt_v_per_soc=np.array(samples["ocv_tilt_v_per_soc"][:n]),
    )


def summarise(values: FloatArray, percent: float) -> dict[str, float]:
    tail = 0.5 * (100.0 - percent)
    return {
        "median": float(np.median(values)),
        "low": float(np.percentile(values, tail)),
        "high": float(np.percentile(values, 100.0 - tail)),
    }


def propagate_baseline(pack: PackBatch, cached: dict, config: ProjectConfig) -> dict[str, object]:
    packs = resampled_packs(pack, cached["samples"], config)
    n = packs.n_packs
    percent = config.identification.ci_percent
    table: dict[str, object] = {"n_resamples": n, "interval_percent": percent}
    for technique in Technique:
        spec = config.technique(technique)
        result = simulate(
            technique,
            packs,
            np.full(n, spec.component.mid),
            np.full(n, spec.frequency_hz.mid),
            config,
        )
        table[technique.key] = {
            "balancing_time_s": summarise(result.balancing_time_s, percent),
            "energy_dissipated_j": summarise(result.energy_dissipated_j, percent),
            "efficiency": summarise(result.efficiency, percent),
            "all_balanced": bool(result.balanced.all()),
        }
    return table
