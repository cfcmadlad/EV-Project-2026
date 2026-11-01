from scripts.claims_base import CheckedClaim, at_most, boolean, is_true, percent
from src.config import Technique
from src.dataset import TARGET_NAMES

TARGET_LABELS = {"balancing_time_s": "balancing time", "energy_dissipated_j": "dissipated energy"}


def abstaining_claims() -> list[CheckedClaim]:
    claims = []
    for name in TARGET_NAMES:
        for technique in Technique:
            claims.append(
                CheckedClaim(
                    f"Normalised conformal 90 % interval on {TARGET_LABELS[name]} for "
                    f"{technique.label.lower()} keeps nominal test coverage",
                    f"technique_selector.abstaining.calibration.{name}.techniques."
                    f"{technique.key}.pass",
                    "coverage 87 to 93 %",
                    is_true,
                    boolean,
                    "single calibration split, 400 test rows per technique",
                )
            )
    base = "technique_selector.abstaining.by_budget.1.policies.conformal_abstain"
    claims += [
        CheckedClaim(
            "Abstaining selector misses a 1 h budget on at most 5 % of the packs where some "
            "technique meets it (300 held-out packs)",
            f"{base}.budget_violation_fraction_among_feasible",
            "<= 5 %",
            at_most(0.05),
            percent,
        ),
        CheckedClaim(
            "Abstaining selector picks the oracle technique for a 1 h budget",
            f"{base}.agrees_with_oracle",
            ">= 90 %",
            lambda v: float(v) >= 0.9,
            percent,
        ),
    ]
    return claims
