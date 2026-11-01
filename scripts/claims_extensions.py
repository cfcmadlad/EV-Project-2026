from scripts.claims_base import CheckedClaim, at_most, boolean, is_true, percent, scientific
from src.config import CONFIG, Technique
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


def thermal_claims() -> list[CheckedClaim]:
    base = "thermal"
    return [
        CheckedClaim(
            "Heat in the thermal model equals the bleed resistor energy booked in the ledger "
            "(passive baseline, continuous and duty-cycled)",
            f"{base}.heat_vs_ledger.resistor_max_relative_gap",
            "<= 1 %",
            at_most(0.01),
            percent,
        ),
        CheckedClaim(
            "Heat in the thermal model equals the cell R0 loss booked in the ledger (baseline, "
            "all three techniques and duty-cycled passive)",
            f"{base}.heat_vs_ledger.cell_max_relative_gap",
            "<= 1 %",
            at_most(0.01),
            percent,
        ),
        CheckedClaim(
            "The thermal integrator closes its own energy balance (stored plus rejected equals "
            "heat in, worst pack)",
            f"{base}.heat_vs_ledger.max_storage_balance_error",
            "<= 1 %",
            at_most(0.01),
            percent,
        ),
        CheckedClaim(
            "Continuous passive bleeding at the baseline stays under the resistor temperature "
            "rise ceiling",
            f"{base}.passive_limited.unconstrained_exceeds_ceiling",
            "peak rise <= 20 K",
            lambda v: v is False,
            boolean,
            "assumed thermal parameters; FAIL means the ceiling is exceeded",
        ),
        CheckedClaim(
            "Thermally aware selector misses a 1 h budget on at most 5 % of the packs where some "
            "technique meets it (thermally limited ground truth, 300 packs)",
            f"{base}.selector.by_budget.1.policies.thermal_selector."
            "budget_violation_fraction_among_feasible",
            "<= 5 %",
            at_most(0.05),
            percent,
        ),
    ]


def shift_claims() -> list[CheckedClaim]:
    claims = []
    for key, label in (
        ("selector", "Boosting selector"),
        ("conformal_abstain", "Hybrid abstaining selector"),
    ):
        claims.append(
            CheckedClaim(
                f"{label} misses a 1 h budget on at most 5 % of the packs where some technique "
                "meets it when the spread, capacity and resistance ranges are 1.25 times the "
                "training ranges (300 packs, no retraining)",
                f"range_shift.by_factor.1p25.violation_by_budget.{key}.1",
                "<= 5 %",
                at_most(0.05),
                percent,
            )
        )
    return claims


def differentiable_claims() -> list[CheckedClaim]:
    claims = []
    for technique in (Technique.BUCK_BOOST, Technique.FLYING_CAPACITOR):
        label = technique.label.lower()
        claims.append(
            CheckedClaim(
                f"The PyTorch twin reproduces the NumPy {label} step (200 random packs, 1000 "
                "steps, SOC within 1e-6 and cumulative energies within 1e-4 relative)",
                f"differentiable_control.parity.{technique.key}.pass",
                "true",
                is_true,
                boolean,
            )
        )
        for n_cells in CONFIG.mpc.cells:
            base = f"differentiable_control.results.{technique.key}.{n_cells}"
            where = f"{label}, {n_cells} cells, Mixed 1 and Mixed 7, 300 packs"
            claims += [
                CheckedClaim(
                    f"The twin plant reproduces the NumPy rule's delivered Ah ({where})",
                    f"{base}.plant_consistency_max_relative",
                    "<= 1e-4 relative",
                    at_most(1e-4),
                    scientific,
                ),
                CheckedClaim(
                    f"Gradient MPC delivers at least as many Ah as the rule in at least 90 % of "
                    f"packs ({where})",
                    f"{base}.win_fraction",
                    ">= 90 %",
                    lambda v: float(v) >= 0.9,
                    percent,
                    "a win is within 1e-3 Ah of the rule or better",
                ),
            ]
        claims.append(
            CheckedClaim(
                f"Gradient MPC closes at least 25 % of the rule-to-reference gap ({label}, "
                "32 cells)",
                f"differentiable_control.results.{technique.key}.32.gap_closed",
                ">= 25 %",
                lambda v: float(v) >= 0.25,
                percent,
                "reference is the lossless ideal-transfer bound",
            )
        )
    return claims
