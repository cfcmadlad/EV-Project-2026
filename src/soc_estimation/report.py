import json
from pathlib import Path

from src.config import ProjectConfig
from src.fingerprint import text_sha256
from src.soc_estimation.data import SPLITS_PATH, TEMPERATURE_PATH

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "soc_estimation_results.json"
)
NOT_RUN = {"status": "not_run"}
STALE = {"status": "stale_cache"}
PAIRS = (
    ("gru", "ekf"),
    ("gru", "cc"),
    ("ekf", "cc"),
    ("ekf2", "ekf"),
    ("ekf2", "gru"),
    ("ekf2", "cc"),
)
COMPARISONS = {
    "nominal_inputs": {
        "cc": "offset_injected",
        "ekf": "nominal",
        "ekf2": "nominal",
        "gru": "nominal",
    },
    "nominal_inputs_cc_at_tester_accuracy": {
        "cc": "offset_tester_accuracy",
        "ekf": "nominal",
        "ekf2": "nominal",
        "gru": "nominal",
    },
    "injected_offset_for_all": {
        "cc": "offset_injected",
        "ekf": "offset_injected",
        "ekf2": "offset_injected",
        "gru": "offset_injected",
    },
}
SETS = ("test_a", "test_b")


def current_inputs() -> dict[str, str]:
    return {
        "soc_splits_sha256": text_sha256(SPLITS_PATH),
        "hg2_temperature_sha256": text_sha256(TEMPERATURE_PATH),
    }


def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    payload = json.loads(RESULTS_PATH.read_text())
    expected = current_inputs()
    if any(payload["inputs"].get(k) != v for k, v in expected.items()):
        return dict(STALE)
    return payload


def strip(node: object) -> object:
    if isinstance(node, dict):
        return {
            k: strip(v) for k, v in node.items() if k not in ("per_cycle_rmse_percent", "per_seed")
        }
    return node


def compare(results: dict, scenario: dict[str, str]) -> dict[str, dict[str, object]]:
    def stat(name: str, test_set: str) -> tuple[float, float]:
        leaf = results[name][scenario[name]][test_set]["all"]
        return leaf["rmse_percent"], leaf.get("rmse_percent_std", 0.0)

    comparisons: dict[str, dict[str, object]] = {}
    for first, second in PAIRS:
        if first not in results or second not in results:
            continue
        row: dict[str, object] = {}
        margins, spreads = [], []
        for test_set in SETS:
            a, a_std = stat(first, test_set)
            b, b_std = stat(second, test_set)
            spread = max(a_std, b_std)
            row[f"rmse_{first}_{test_set}"] = a
            row[f"rmse_{second}_{test_set}"] = b
            row[f"seed_spread_{test_set}"] = spread
            row[f"margin_{test_set}"] = b - a
            margins.append(b - a)
            spreads.append(spread)
        wins_first = all(m > s for m, s in zip(margins, spreads, strict=True))
        wins_second = all(-m > s for m, s in zip(margins, spreads, strict=True))
        row["verdict"] = (
            f"{first}_beats_{second}"
            if wins_first
            else f"{second}_beats_{first}"
            if wins_second
            else "no_verdict"
        )
        comparisons[f"{first}_vs_{second}"] = row
    return comparisons


def soc_estimation_block(config: ProjectConfig) -> dict[str, object]:
    if not SPLITS_PATH.exists() or not TEMPERATURE_PATH.exists():
        return dict(NOT_RUN)
    payload = load_results()
    if payload is None:
        return dict(NOT_RUN)
    if payload.get("status") == "stale_cache":
        return dict(STALE)
    splits = json.loads(SPLITS_PATH.read_text())
    results = payload["results"]
    return {
        "status": "ok",
        "reference_definition": splits["definition"]["reference_soc"],
        "label_note": payload["label_note"],
        "splits": payload["splits_summary"],
        "rejected_files": {r["file"]: r["reason"] for r in splits["rejected_files"]},
        "absent_or_unusable": splits["absent_or_unusable"],
        "ekf": payload["ekf"],
        "ekf2": payload["ekf2"],
        "gru": {k: v for k, v in payload["gru"].items() if k not in ("training_time_s",)},
        "results": strip(results),
        "comparisons": {name: compare(results, scenario) for name, scenario in COMPARISONS.items()},
        "feedthrough": payload["feedthrough"],
    }


def timing_block() -> dict[str, object] | None:
    payload = load_results()
    if payload is None or payload.get("status") == "stale_cache":
        return None
    return {
        "gru_training_time_s": payload["gru"]["training_time_s"],
        "gru_weight_bytes_max": payload["gru"]["weight_bytes_max"],
        "gru_parameters": payload["gru"]["parameters"],
    }
