import numpy as np

from src.balancing import simulate
from src.baselines import physics_prediction
from src.config import FloatArray, ProjectConfig, Technique
from src.dataset import pack_features
from src.pack import PackBatch, sample_packs
from src.params import SECONDS_PER_HOUR
from src.predictor import PredictorReport

POLICY_KEYS = (
    "selector",
    "physics_selector",
    "oracle",
    "always_passive",
    "always_buck_boost",
    "always_flying_capacitor",
)


def budget_key(hours: float) -> str:
    return f"{hours:g}".replace(".", "p")


def mid_values(config: ProjectConfig, n: int) -> tuple[list[FloatArray], list[FloatArray]]:
    return (
        [np.full(n, config.technique(t).component.mid) for t in Technique],
        [np.full(n, config.technique(t).frequency_hz.mid) for t in Technique],
    )


def simulate_all(config: ProjectConfig, packs: PackBatch) -> tuple[FloatArray, FloatArray]:
    components, frequencies = mid_values(config, packs.n_packs)
    runs = [simulate(t, packs, components[t], frequencies[t], config) for t in Technique]
    return (
        np.column_stack([r.balancing_time_s for r in runs]),
        np.column_stack([r.energy_dissipated_j for r in runs]),
    )


def surrogate_predictions(
    config: ProjectConfig, packs: PackBatch, report: PredictorReport
) -> tuple[FloatArray, FloatArray]:
    components, frequencies = mid_values(config, packs.n_packs)
    time_cols, energy_cols = [], []
    for t in Technique:
        features = pack_features(t, packs, components[t], frequencies[t])
        time_cols.append(report.models[0].predict(features))
        energy_cols.append(report.models[1].predict(features))
    return np.column_stack(time_cols), np.column_stack(energy_cols)


def physics_predictions(config: ProjectConfig, packs: PackBatch) -> tuple[FloatArray, FloatArray]:
    components, frequencies = mid_values(config, packs.n_packs)
    runs = [
        physics_prediction(
            t, packs.soc0, packs.capacity_ah, packs.r0_ohm, components[t], frequencies[t], config
        )
        for t in Technique
    ]
    return (
        np.column_stack([r.balancing_time_s for r in runs]),
        np.column_stack([r.energy_dissipated_j for r in runs]),
    )


def choose(time_s: FloatArray, energy_j: FloatArray, budget_s: float) -> FloatArray:
    feasible = time_s <= budget_s
    cheapest = np.argmin(np.where(feasible, energy_j, np.inf), axis=1)
    fastest = np.argmin(time_s, axis=1)
    return np.where(feasible.any(axis=1), cheapest, fastest)


def policy_row(
    choice: FloatArray,
    true_time: FloatArray,
    true_energy: FloatArray,
    oracle: FloatArray,
    budget_s: float,
) -> dict[str, float]:
    rows = np.arange(choice.size)
    energy = true_energy[rows, choice]
    feasible = (true_time <= budget_s).any(axis=1)
    return {
        "budget_violation_fraction_among_feasible": float(
            np.mean(true_time[rows, choice][feasible] > budget_s)
        ),
        "agrees_with_oracle": float(np.mean(choice == oracle)),
        "budget_violation_fraction": float(np.mean(true_time[rows, choice] > budget_s)),
        "mean_energy_j": float(energy.mean()),
        "mean_regret_vs_oracle_j": float(np.mean(energy - true_energy[rows, oracle])),
        "saving_vs_always_passive_j": float(np.mean(true_energy[:, Technique.PASSIVE] - energy)),
        "saving_vs_always_buck_boost_j": float(
            np.mean(true_energy[:, Technique.BUCK_BOOST] - energy)
        ),
    }


def selector_block(
    config: ProjectConfig, rng: np.random.Generator, report: PredictorReport
) -> dict[str, object]:
    spec = config.selector
    packs = sample_packs(spec.packs, rng, config.cell, config.pack)
    true_time, true_energy = simulate_all(config, packs)
    surrogate_time, surrogate_energy = surrogate_predictions(config, packs, report)
    physics_time, physics_energy = physics_predictions(config, packs)
    budgets: dict[str, object] = {}
    n = packs.n_packs
    for hours in spec.budgets_h:
        budget_s = hours * SECONDS_PER_HOUR
        oracle = choose(true_time, true_energy, budget_s)
        choices = {
            "selector": choose(surrogate_time, surrogate_energy, budget_s),
            "physics_selector": choose(physics_time, physics_energy, budget_s),
            "oracle": oracle,
            **{f"always_{t.key}": np.full(n, int(t)) for t in Technique},
        }
        budgets[budget_key(hours)] = {
            "oracle_feasible_fraction": float(np.mean((true_time <= budget_s).any(axis=1))),
            "policies": {
                key: policy_row(choice, true_time, true_energy, oracle, budget_s)
                for key, choice in choices.items()
            },
        }
    return {
        "status": "ok",
        "n_packs": n,
        "budgets_h": list(spec.budgets_h),
        "components": "mid-range component values for every technique",
        "surrogate": "gradient boosting models of section h, trained on the 80 % training split",
        "min_accuracy": spec.min_accuracy,
        "max_violation": spec.max_violation,
        "by_budget": budgets,
    }
