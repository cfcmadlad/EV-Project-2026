from dataclasses import replace

import numpy as np

from src.abstain import abstaining_selector_block
from src.config import FloatArray, ProjectConfig
from src.ml_eval import ConformalSplit, EvalData, FittedModel
from src.predictor import PredictorReport
from src.selector import SelectorInputs, budget_key, selector_block, selector_inputs


def scaled_config(config: ProjectConfig, factor: float) -> ProjectConfig:
    pack = config.pack
    widened = replace(
        pack,
        soc_spread_max=pack.soc_spread_max * factor,
        capacity_tolerance=pack.capacity_tolerance * factor,
        r0_tolerance=pack.r0_tolerance * factor,
    )
    return replace(
        config,
        params=replace(config.params, pack=widened),
        selector=replace(config.selector, packs=config.shift.packs),
    )


def mdape(prediction: FloatArray, truth: FloatArray) -> float:
    return float(np.median(np.abs(prediction - truth) / truth))


def factor_key(factor: float) -> str:
    return f"{factor:g}".replace(".", "p")


def shift_block(
    config: ProjectConfig,
    rng: np.random.Generator,
    report: PredictorReport,
    data: EvalData,
    split: ConformalSplit,
    fitted: dict[tuple[str, int], FittedModel],
    best: dict[str, str],
    fixed: dict[str, object],
) -> dict[str, object]:
    spec = config.shift
    factors = (1.0, *spec.factors)
    streams = rng.spawn(len(factors))
    rows: dict[str, object] = {}
    for factor, stream in zip(factors, streams, strict=True):
        shifted = scaled_config(config, factor)
        inputs: SelectorInputs = selector_inputs(shifted, stream, report)
        boosting = selector_block(shifted, inputs)
        hybrid = abstaining_selector_block(data, shifted, split, fitted, best, fixed, inputs)
        budget = budget_key(spec.budget_h)
        rows[factor_key(factor)] = {
            "factor": factor,
            "boosting_time_mdape": mdape(inputs.surrogate_time, inputs.true_time),
            "boosting_energy_mdape": mdape(inputs.surrogate_energy, inputs.true_energy),
            "hybrid_time_mdape": hybrid["time_mdape"],
            "hybrid_energy_mdape": hybrid["energy_mdape"],
            "hybrid_time_interval_coverage": hybrid["time_interval_coverage"],
            "oracle_feasible_fraction": boosting["by_budget"][budget]["oracle_feasible_fraction"],
            "violation_by_budget": {
                "selector": {
                    key: value["policies"]["selector"]["budget_violation_fraction_among_feasible"]
                    for key, value in boosting["by_budget"].items()
                },
                "conformal_abstain": {
                    key: value["policies"]["conformal_abstain"][
                        "budget_violation_fraction_among_feasible"
                    ]
                    for key, value in hybrid["by_budget"].items()
                },
            },
            "agreement_at_budget": {
                "selector": boosting["by_budget"][budget]["policies"]["selector"][
                    "agrees_with_oracle"
                ],
                "conformal_abstain": hybrid["by_budget"][budget]["policies"]["conformal_abstain"][
                    "agrees_with_oracle"
                ],
            },
        }
    return {
        "status": "ok",
        "factors": list(factors),
        "n_packs_per_factor": spec.packs,
        "budget_h": spec.budget_h,
        "max_violation": spec.max_violation,
        "widened": "soc spread upper limit, capacity tolerance and r0 tolerance times the factor",
        "by_factor": rows,
        "first_factor_above_limit": {
            policy: next(
                (
                    f
                    for f in factors
                    if rows[factor_key(f)]["violation_by_budget"][policy][budget_key(spec.budget_h)]
                    > spec.max_violation
                ),
                None,
            )
            for policy in ("selector", "conformal_abstain")
        },
    }
