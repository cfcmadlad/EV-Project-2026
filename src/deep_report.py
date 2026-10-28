import json

from src.config import PROJECT_ROOT
from src.fingerprint import maps_sha256

PROCESSED = PROJECT_ROOT / "data" / "processed"
NOT_RUN = {"status": "not_run"}
STALE = {"status": "stale_cache"}
CACHES = {
    "surrogates": "deep_surrogates.json",
    "trajectory": "trajectory_results.json",
    "soc": "soc_deep_results.json",
    "rl": "rl2_results.json",
}


def _load(name: str) -> dict | None:
    path = PROCESSED / CACHES[name]
    return json.loads(path.read_text()) if path.is_file() else None


def _strip_timing(node: object) -> object:
    if isinstance(node, dict):
        return {k: _strip_timing(v) for k, v in node.items() if k != "timing"}
    return node


SURROGATE_PAIRS = (
    ("mlp", "hgb"),
    ("cell_gru", "hgb"),
    ("cell_transformer", "hgb"),
    ("cell_gru", "mlp"),
    ("cell_transformer", "cell_gru"),
    ("mlp_physics", "hgb_physics"),
    ("cell_gru_physics", "hgb_physics"),
    ("cell_gru_physics", "hgb"),
    ("cell_gru_physics", "mlp_physics"),
)
GROUPS = ("overall", "flying_capacitor")
TARGETS = ("balancing_time_s", "energy_dissipated_j")


def surrogate_verdicts(models: dict) -> dict[str, dict[str, dict[str, bool]]]:
    out: dict[str, dict[str, dict[str, bool]]] = {}
    for first, second in SURROGATE_PAIRS:
        verdict: dict[str, dict[str, bool]] = {}
        for target in TARGETS:
            verdict[target] = {}
            for group in GROUPS:
                a = models[first]["metrics"][target][group]["r2"]
                b = models[second]["metrics"][target][group]["r2"]
                verdict[target][group] = bool(a["min"] > b["max"])
        out[f"{first}_vs_{second}"] = verdict
    return out


def shape_gaps(trajectory: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, model in trajectory["models"].items():
        out[name] = {}
        for tech in ("passive", "buck_boost", "flying_capacitor"):
            leaf = model["metrics"][tech]
            out[name][tech] = abs(
                leaf["half_spread_time_pred"]["mean"] - leaf["half_spread_time_true"]["mean"]
            )
    return out


def rl_summary(cache: dict) -> dict[str, object]:
    out: dict[str, object] = {"timesteps": cache["timesteps"]}
    for size, cycles in cache["test"].items():
        out[size] = {}
        for variant in next(iter(cycles.values()))["vs_rule"]:
            per_cycle = {c: v["vs_rule"][variant] for c, v in cycles.items()}
            out[size][variant] = {
                "beats_rule_on_every_cycle": all(
                    v["exceeds_standard_errors"] for v in per_cycle.values()
                ),
                "worst_cycle_gain": min(
                    v["mean_improvement_over_rule"] for v in per_cycle.values()
                ),
                "mean_gain": sum(v["mean_improvement_over_rule"] for v in per_cycle.values())
                / len(per_cycle),
            }
        out[size]["pack_ah"] = {
            c: {
                **{k: v["table"][k]["pack_ah"] for k in ("none", "rule_based", "greedy")},
                **{
                    variant: sum(v["table"][f"{variant}_s{seed}"]["pack_ah"] for seed in (0, 1, 2))
                    / 3.0
                    for variant in v["vs_rule"]
                },
                "ideal_reference": v["table"]["ideal_reference"]["pack_ah"],
            }
            for c, v in cycles.items()
        }
    return out


def deep_models_block(dataset_hash: str) -> dict[str, object]:
    out: dict[str, object] = {}
    surrogates = _load("surrogates")
    if surrogates is None:
        out["surrogates"] = dict(NOT_RUN)
    elif surrogates.get("dataset_sha256") != dataset_hash:
        out["surrogates"] = dict(STALE)
    else:
        out["surrogates"] = (
            {"status": "ok"}
            | _strip_timing(surrogates)
            | {"verdicts": surrogate_verdicts(surrogates["models"])}
        )
    trajectory = _load("trajectory")
    if trajectory is not None and trajectory.get("maps_sha256") != maps_sha256():
        out["trajectory"] = dict(STALE)
    else:
        out["trajectory"] = (
            dict(NOT_RUN)
            if trajectory is None
            else {"status": "ok"} | _strip_timing(trajectory) | {"shape": shape_gaps(trajectory)}
        )
    soc = _load("soc")
    out["soc"] = dict(NOT_RUN) if soc is None else {"status": "ok"} | _soc_summary(soc)
    rl = _load("rl")
    out["rl"] = dict(NOT_RUN) if rl is None else {"status": "ok"} | rl_summary(rl)
    return out


def soc_verdicts(results: dict) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for winner, loser in (("ekf_gru_hybrid", "ekf2"), ("ekf_gru_hybrid", "gru_augmented")):
        out[f"{winner}_beats_{loser}"] = {}
        for split in ("test_a", "test_b"):
            a = results[winner]["nominal"][split]
            b = results[loser]["nominal"][split]
            margin = a.get("rmse_percent_std", 0.0) + b.get("rmse_percent_std", 0.0)
            out[f"{winner}_beats_{loser}"][split] = bool(
                a["rmse_percent"] + margin < b["rmse_percent"]
            )
    return out


def _soc_summary(soc: dict) -> dict[str, object]:
    quant = soc["quantisation"]
    return {
        "selected_window_s": soc["selected_window_s"],
        "parameters": soc["parameters"],
        "macs_per_estimate": soc["macs_per_estimate"],
        "results": {
            k: {
                sc: {
                    s: {m: v[m] for m in ("rmse_percent", "mae_percent", "max_abs_percent")}
                    | {m + "_std": v.get(m + "_std", 0.0) for m in ("rmse_percent",)}
                    for s, v in by_set.items()
                }
                for sc, by_set in scenarios.items()
            }
            for k, scenarios in soc["results"].items()
        },
        "quantisation": quant,
        "verdicts": soc_verdicts(soc["results"]),
        "int8_rmse_ratio": {
            k: quant[k]["8"]["test_rmse_percent"] / quant[k]["32"]["test_rmse_percent"]
            for k in quant
        },
        "feedthrough": soc["feedthrough"]["techniques"],
    }


def deep_timing() -> dict[str, object]:
    out: dict[str, object] = {}
    for name in ("surrogates", "trajectory"):
        cache = _load(name)
        if cache is not None:
            out[name] = {m: v["timing"] for m, v in cache["models"].items()}
    return out
