from scripts.claims_base import CheckedClaim, at_most, boolean, is_true, percent
from src.deep_report import SURROGATE_PAIRS

BASE = "deep_models"
TIME = "balancing_time_s"
ENERGY = "energy_dissipated_j"
TECHNIQUES = ("passive", "buck_boost", "flying_capacitor")
LABELS = {
    "hgb": "gradient boosting",
    "hgb_physics": "gradient boosting on the physics residual",
    "mlp": "MLP",
    "mlp_physics": "MLP on the physics residual",
    "cell_gru": "cell-sequence GRU",
    "cell_gru_physics": "cell-sequence GRU on the physics residual",
    "cell_transformer": "cell-token Transformer",
}


def at_least(limit: float):
    return lambda value: float(value) >= limit


def number(value: object) -> str:
    return f"{float(value):.4g}"


def surrogate_claims() -> list[CheckedClaim]:
    claims = []
    note = "all 3 seeds: worst seed of the first above the best seed of the second"
    for first, second in SURROGATE_PAIRS:
        for target, group, label in (
            (TIME, "overall", "balancing-time R2"),
            (TIME, "flying_capacitor", "flying-capacitor balancing-time R2"),
            (ENERGY, "overall", "energy R2"),
        ):
            claims.append(
                CheckedClaim(
                    f"The {LABELS[first]} beats {LABELS[second]} on {label}",
                    f"{BASE}.surrogates.verdicts.{first}_vs_{second}.{target}.{group}",
                    "true",
                    is_true,
                    boolean,
                    note,
                )
            )
    for model in ("hgb", "cell_gru_physics"):
        claims.append(
            CheckedClaim(
                f"The {LABELS[model]} reaches a balancing-time R2 of 0.99 (mean of 3 seeds)",
                f"{BASE}.surrogates.models.{model}.metrics.{TIME}.overall.r2.mean",
                ">= 0.99",
                at_least(0.99),
                number,
            )
        )
    return claims


def trajectory_claims() -> list[CheckedClaim]:
    claims = []
    for model in ("gru_decoder", "transformer_decoder"):
        name = model.replace("_", " ")
        claims.append(
            CheckedClaim(
                f"The {name} predicts the time to a 1 % spread within 10 % (median error, "
                "mean of 3 seeds)",
                f"{BASE}.trajectory.models.{model}.metrics.overall."
                "time_to_1pct_median_rel_error_percent.mean",
                "<= 10",
                at_most(10.0),
                number,
            )
        )
        claims.append(
            CheckedClaim(
                f"The {name} predicts the SOC spread trajectory within 0.5 percentage points "
                "(RMSE, mean of 3 seeds)",
                f"{BASE}.trajectory.models.{model}.metrics.overall.spread_rmse_percent.mean",
                "<= 0.5",
                at_most(0.5),
                number,
            )
        )
        for tech in TECHNIQUES:
            claims.append(
                CheckedClaim(
                    f"The {name} reproduces the {tech.replace('_', ' ')} trajectory shape "
                    "(time to half the initial spread within 0.05 of the normalised time)",
                    f"{BASE}.trajectory.shape.{model}.{tech}",
                    "<= 0.05",
                    at_most(0.05),
                    number,
                )
            )
    return claims


def soc_claims() -> list[CheckedClaim]:
    claims = []
    for loser, label in (("ekf2", "2RC EKF"), ("gru_augmented", "augmented GRU")):
        for split in ("test_a", "test_b"):
            claims.append(
                CheckedClaim(
                    f"The hybrid EKF plus GRU beats the {label} on nominal RMSE "
                    f"({split.replace('_', ' ')}, margin above the seed spread)",
                    f"{BASE}.soc.verdicts.ekf_gru_hybrid_beats_{loser}.{split}",
                    "true",
                    is_true,
                    boolean,
                )
            )
    for model in ("gru_augmented", "ekf_gru_hybrid"):
        claims.append(
            CheckedClaim(
                f"int8 weights keep the {model.replace('_', ' ')} RMSE within 5 % of float32",
                f"{BASE}.soc.int8_rmse_ratio.{model}",
                "<= 1.05",
                at_most(1.05),
                number,
            )
        )
    for tech in ("passive", "buck_boost"):
        claims.append(
            CheckedClaim(
                f"Balancing on the hybrid SOC ({tech.replace('_', ' ')}) brings most packs to "
                "the 1 % true spread target",
                f"{BASE}.soc.feedthrough.{tech}.ekf_gru_hybrid.fraction_true_spread_above_tolerance",
                "<= 50 %",
                at_most(0.5),
                percent,
            )
        )
    return claims


def rl_claims() -> list[CheckedClaim]:
    claims = []
    for size, label in (("n4", "4 cells"), ("n8", "8 cells")):
        for variant in (
            "residual_plain",
            "residual_shaped",
            "imitation_plain",
            "imitation_shaped",
        ):
            claims.append(
                CheckedClaim(
                    f"The {variant.replace('_', ' ')} PPO policy beats the rule-based controller "
                    f"on the reward on every test cycle ({label}, gain above two standard errors)",
                    f"{BASE}.rl.{size}.{variant}.beats_rule_on_every_cycle",
                    "true",
                    is_true,
                    boolean,
                    "3 seeds, mean paired gain per pack",
                )
            )
    return claims


def deep_claims() -> list[CheckedClaim]:
    return surrogate_claims() + trajectory_claims() + soc_claims() + rl_claims()
