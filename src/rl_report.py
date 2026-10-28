import json
from pathlib import Path

from src.config import ProjectConfig
from src.drive_cycle import PROFILES_PATH
from src.fingerprint import binary_sha256, text_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "processed" / "rl_results.json"
POLICY_DIR = ROOT / "data" / "processed" / "rl_policy"
BASELINES = ("none", "rule_based", "greedy")


def load_rl_results() -> dict | None:
    if not RESULTS_PATH.exists() or not PROFILES_PATH.exists():
        return None
    payload = json.loads(RESULTS_PATH.read_text())
    inputs = payload["inputs"]
    fresh = inputs["drive_profiles_sha256"] == text_sha256(PROFILES_PATH) and all(
        (POLICY_DIR / name).exists() and binary_sha256(POLICY_DIR / name) == value
        for name, value in inputs["policies_sha256"].items()
    )
    return payload if fresh else {"status": "stale_cache"}


def verdicts(test: dict[str, dict]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for other in BASELINES:
        row: dict[str, object] = {}
        for field in ("reward", "pack_ah"):
            cycles = {c: block["ppo_vs"][other][field] for c, block in test.items()}
            row[field] = {
                "rl_beats": all(
                    v["exceeds_standard_errors"] and v["mean_improvement"] > 0.0
                    for v in cycles.values()
                ),
                "mean_improvement_by_cycle": {c: v["mean_improvement"] for c, v in cycles.items()},
                "standard_error_by_cycle": {c: v["standard_error"] for c, v in cycles.items()},
            }
        out[other] = row
    return out


def rl_block(config: ProjectConfig) -> dict[str, object]:
    payload = load_rl_results()
    if payload is None:
        return {"status": "not_run"}
    if payload.get("status") == "stale_cache":
        return {"status": "stale_cache"}
    test = payload["test"]
    outside = max(
        v["outside_s"]["mean"] * config.rl.test_packs
        for block in test.values()
        for name, v in block.items()
        if name != "ppo_vs" and name != "ppo_mean_of_seeds" and isinstance(v, dict)
    )
    unfinished = min(
        v["finished"]["mean"]
        for block in test.values()
        for name, v in block.items()
        if name not in ("ppo_vs", "ppo_mean_of_seeds")
    )
    return {
        "status": "ok",
        "design": {
            "control_step_s": config.rl.control_step_s,
            "d_max": config.rl.d_max,
            "loss_weight_ah_per_j": config.rl.loss_weight_ah_per_j,
            "estimate_sigma": config.rl.estimate_sigma,
            "train_cycles": list(config.rl.train_cycles),
            "test_cycles": list(config.rl.test_cycles),
            "test_packs": config.rl.test_packs,
            "standard_errors": config.rl.standard_errors,
        },
        "training": {
            k: v for k, v in payload["training"].items() if k not in ("training_time_s", "curves")
        },
        "test": test,
        "verdicts": verdicts(test),
        "limit_violation_seconds_total": float(outside),
        "all_episodes_finished": bool(unfinished >= 1.0),
    }


def timing_block() -> dict[str, object] | None:
    payload = load_rl_results()
    if payload is None or payload.get("status") == "stale_cache":
        return None
    return {"training_time_s": payload["training"]["training_time_s"]}
