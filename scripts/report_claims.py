import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
CLAIMS_PATH = ROOT / "results" / "claims.md"
TECHNIQUES = (
    ("passive", "passive shunt"),
    ("buck_boost", "inductor buck-boost"),
    ("flying_capacitor", "flying capacitor"),
)
LINK_NOTE = (
    "averaged link equations book each cell's R0 drop per link, not on the net cell current; "
    "see README"
)
HEADER = (
    "| Claim | Evidence key | Value | Threshold | Status | Note |",
    "|---|---|---|---|---|---|",
)


@dataclass(frozen=True, slots=True)
class CheckedClaim:
    text: str
    evidence: str
    threshold: str
    passes: Callable[[object], bool]
    render: Callable[[object], str]
    note: str = ""


@dataclass(frozen=True, slots=True)
class UnsupportedClaim:
    text: str
    reason: str


def percent(value: object) -> str:
    return f"{float(value) * 100.0:.4g} %"


def millivolts(value: object) -> str:
    return f"{float(value) * 1000.0:.4g} mV"


def scientific(value: object) -> str:
    return f"{float(value):.2e}"


def boolean(value: object) -> str:
    return str(bool(value)).lower()


def at_most(limit: float) -> Callable[[object], bool]:
    return lambda value: float(value) <= limit


def abs_at_most(limit: float) -> Callable[[object], bool]:
    return lambda value: abs(float(value)) <= limit


def between(low: float, high: float) -> Callable[[object], bool]:
    return lambda value: low <= float(value) <= high


def positive(value: object) -> bool:
    return float(value) > 0.0


def signed_percent(value: object) -> str:
    return f"{float(value) * 100.0:+.4g} %"


def is_true(value: object) -> bool:
    return value is True


def validation_claims() -> list[CheckedClaim]:
    claims = []
    for cycle in ("UDDS", "LA92"):
        for field, label, limit in (
            ("rmse_whole_v", "whole cycle", 0.030),
            ("rmse_window_v", "20 to 80 % SOC window", 0.020),
        ):
            claims.append(
                CheckedClaim(
                    f"Fitted 2RC cell model (HPPC only) reproduces measured {cycle} terminal "
                    f"voltage at 25 C, {label} RMSE, no retuning",
                    f"cell_validation.validation.{cycle}.2rc.{field}",
                    f"<= {limit * 1000:.0f} mV",
                    at_most(limit),
                    millivolts,
                )
            )
    claims.append(
        CheckedClaim(
            "Single-branch equivalent of the 2RC model (the form the balancing simulator can "
            "use) reproduces measured UDDS voltage, whole cycle RMSE",
            "cell_validation.validation.UDDS.lumped.rmse_whole_v",
            "<= 30 mV",
            at_most(0.030),
            millivolts,
            "simulator has one RC branch; the validated model has two",
        )
    )
    claims.append(
        CheckedClaim(
            "Single-branch equivalent of the 2RC model reproduces measured LA92 voltage, "
            "whole cycle RMSE",
            "cell_validation.validation.LA92.lumped.rmse_whole_v",
            "<= 30 mV",
            at_most(0.030),
            millivolts,
            "simulator has one RC branch; the validated model has two",
        )
    )
    claims.append(
        CheckedClaim(
            "A 1RC model fitted to the same HPPC data meets the UDDS whole cycle limit",
            "cell_validation.validation.UDDS.1rc.rmse_whole_v",
            "<= 30 mV",
            at_most(0.030),
            millivolts,
            "reason the 2RC model was fitted and kept",
        )
    )
    return claims


def converter_claims() -> list[CheckedClaim]:
    note = "LTspice, DC sources with series R0, 6 points (3 operating points x 2 SOC pairs)"
    rows = [
        (
            "Averaged buck-boost mean currents (source and sink cell) match switching-level SPICE",
            "converter_check.buck_boost.max_abs_current_error",
            "<= 5 %",
            0.05,
        ),
        (
            "Averaged buck-boost total dissipated power matches SPICE I^2 R loss",
            "converter_check.buck_boost.max_abs_total_loss_error",
            "<= 15 %",
            0.15,
        ),
        (
            "Two-phase flying-capacitor mean currents match switching-level SPICE",
            "converter_check.flying_capacitor.two_phase.max_abs_current_error",
            "<= 5 %",
            0.05,
        ),
        (
            "Two-phase flying-capacitor total dissipated power matches SPICE I^2 R loss",
            "converter_check.flying_capacitor.two_phase.max_abs_total_loss_error",
            "<= 15 %",
            0.15,
        ),
        (
            "Midsem flying-capacitor current formula (kept as a named option) matches SPICE",
            "converter_check.flying_capacitor.midsem.max_abs_current_error",
            "<= 5 %",
            0.05,
        ),
    ]
    claims = [
        CheckedClaim(text, evidence, threshold, at_most(limit), percent, note)
        for text, evidence, threshold, limit in rows
    ]
    claims.append(
        CheckedClaim(
            "SPICE runs are periodic steady state, step independent, energy conserving, "
            "discontinuous-conduction for the buck-boost and non-overlapping for the capacitor",
            "converter_check.measurement_quality.pass",
            "all five quality checks",
            is_true,
            boolean,
        )
    )
    return claims


MODEL_NAMES = {
    "hgb": "the midsem HGB model",
    "physics": "the physics formula",
    "hybrid": "the hybrid (HGB on the physics residual)",
}
VERDICT_PAIRS = (("hgb", "physics"), ("hybrid", "hgb"), ("hybrid", "physics"))
TARGET_TEXT = (("balancing_time_s", "balancing time"), ("energy_dissipated_j", "energy dissipated"))
VERDICT_NOTE = "MAE lower in all 5 folds and mean gain above the larger fold std"


def ml_eval_claims() -> list[CheckedClaim]:
    base = "ml_eval.physics_baseline_check"
    claims = [
        CheckedClaim(
            "Passive physics baseline reproduces the simulated balancing time on the baseline pack",
            f"{base}.passive.time_error",
            "<= 3 %",
            abs_at_most(0.03),
            signed_percent,
        ),
        CheckedClaim(
            "Passive physics baseline reproduces the simulated energy dissipated on the baseline "
            "pack",
            f"{base}.passive.energy_error",
            "<= 3 %",
            abs_at_most(0.03),
            signed_percent,
        ),
    ]
    for target, target_text in TARGET_TEXT:
        for key, label in TECHNIQUES:
            for better, worse in VERDICT_PAIRS:
                claims.append(
                    CheckedClaim(
                        f"{MODEL_NAMES[better].capitalize()} beats {MODEL_NAMES[worse]} on "
                        f"{target_text} for {label} (5-fold CV)",
                        f"ml_eval.cross_validation.verdicts.{target}.{key}."
                        f"{better}_vs_{worse}.beats",
                        VERDICT_NOTE,
                        is_true,
                        boolean,
                    )
                )
    for target, target_text in TARGET_TEXT:
        for key, label in TECHNIQUES:
            claims.append(
                CheckedClaim(
                    f"Conformal 90 % interval on {target_text} for {label} keeps nominal "
                    "test coverage",
                    f"ml_eval.conformal.{target}.techniques.{key}.coverage",
                    "87 to 93 %",
                    between(0.87, 0.93),
                    percent,
                    "Mondrian by technique, absolute log residual, best model by CV MAE",
                )
            )
    for target, target_text in TARGET_TEXT:
        claims.append(
            CheckedClaim(
                f"The four individual cell SOCs carry information the HGB model uses for "
                f"{target_text} (removing them raises the CV MAE)",
                f"ml_eval.feature_ablation.{target}.cross_validation.mae_change_percent.overall",
                "> 0 %",
                positive,
                lambda value: f"{float(value):+.4g} %",
            )
        )
    return claims


def drive_cycle_claims() -> list[CheckedClaim]:
    scope = "this 4-cell pack under measured Mixed 1 current, one pack, one cycle"
    claims = [
        CheckedClaim(
            "With balancing off, every cell's SOC change equals -integral(I dt)/(3600 Q_i) "
            "under measured drive and charge current",
            "drive_cycle.soc_conservation.max_relative_error",
            "<= 1e-9",
            at_most(1e-9),
            scientific,
        ),
        CheckedClaim(
            "Balancing energy ledger closes under load (net chemical energy removed vs element, "
            "R0 cross-term and RC terms)",
            "drive_cycle.energy_ledger_under_load.max_vs_removed",
            "<= 1 % of E_removed",
            at_most(0.01),
            percent,
        ),
    ]
    for key, label in (
        ("buck_boost", "inductor buck-boost"),
        ("flying_capacitor", "flying capacitor"),
    ):
        claims.append(
            CheckedClaim(
                f"Driving: {label} increases usable pack Ah before the first cell hits a limit "
                f"({scope})",
                f"drive_cycle.scenarios.driving.{key}.gain_vs_none_ah",
                "> 0 Ah",
                positive,
                lambda v: f"{float(v):+.3f} Ah",
            )
        )
    claims.append(
        CheckedClaim(
            f"Driving: passive shunt leaves usable pack Ah unchanged ({scope})",
            "drive_cycle.scenarios.driving.passive.gain_vs_none_ah",
            "|gain| <= 1e-6 Ah",
            abs_at_most(1e-6),
            lambda v: f"{float(v):+.3g} Ah",
            "passive only drains fuller cells; the limiting cell is untouched",
        )
    )
    return claims


def temperature_claims() -> list[CheckedClaim]:
    claims = []
    for temperature in (0, 10, 25, 40):
        for cycle in ("UDDS", "LA92"):
            if temperature == 10 and cycle == "LA92":
                continue
            for field, label, limit in (
                ("rmse_whole_v", "whole cycle", 0.030),
                ("rmse_window_v", "20 to 80 % SOC window", 0.020),
            ):
                claims.append(
                    CheckedClaim(
                        f"Kept fitted model reproduces measured {cycle} at {temperature} C, "
                        f"{label} RMSE (HPPC fit at that temperature only)",
                        f"temperature.per_temperature.{temperature}.validation.{cycle}.{field}",
                        f"<= {limit * 1000:.0f} mV",
                        at_most(limit),
                        millivolts,
                        "10 C LA92 file is unusable (non-monotonic time)"
                        if temperature == 10
                        else "",
                    )
                )
    for temperature in (0, 10, 25, 40):
        for cycle in ("UDDS", "LA92"):
            if temperature == 10 and cycle == "LA92":
                continue
            for field, label, limit in (
                ("rmse_whole_v", "whole cycle", 0.030),
                ("rmse_window_v", "20 to 80 % SOC window", 0.020),
            ):
                claims.append(
                    CheckedClaim(
                        f"Calibrated nonlinear 2RC model reproduces measured {cycle} at "
                        f"{temperature} C, {label} RMSE (tables from HPPC, six scale factors "
                        "fitted on other drive cycles)",
                        f"temperature.per_temperature.{temperature}.improved_model.validation."
                        f"{cycle}.{field}",
                        f"<= {limit * 1000:.0f} mV",
                        at_most(limit),
                        millivolts,
                        "validation cycles never used for fitting; not usable in the simulator",
                    )
                )
    claims += [
        CheckedClaim(
            "R0(T) is described by a log-quadratic form in 1/T (largest error when one "
            "temperature is left out)",
            "temperature.r0_forms.log_quadratic_in_inverse_temperature."
            "leave_one_temperature_out_max_relative_error",
            "<= 10 %",
            at_most(0.10),
            percent,
            "three parameters on four temperatures",
        ),
        CheckedClaim(
            "R0 of the fitted cell follows one Arrhenius law over 0 to 40 C (largest relative "
            "deviation of the fit from the four measured values)",
            "temperature.arrhenius.max_relative_residual",
            "<= 10 %",
            at_most(0.10),
            percent,
            "bootstrap interval on Ea reflects measurement noise only, not this misfit",
        ),
        CheckedClaim(
            "A controller acting on SOC estimates with 1 % SOC bias leaves most buck-boost packs "
            "above the 1 % true spread target",
            "soc_error.techniques.buck_boost.sigma_1pct.fraction_true_spread_above_tolerance",
            "> 50 % of packs",
            lambda v: float(v) > 0.5,
            percent,
            "200 random packs, fixed seed, baseline component values",
        ),
        CheckedClaim(
            "A controller acting on SOC estimates with 1 % SOC bias leaves most passive packs "
            "above the 1 % true spread target",
            "soc_error.techniques.passive.sigma_1pct.fraction_true_spread_above_tolerance",
            "> 50 % of packs",
            lambda v: float(v) > 0.5,
            percent,
            "200 random packs, fixed seed, baseline component values",
        ),
        CheckedClaim(
            "SOC-error study with sigma = 0 reproduces the deterministic simulation exactly",
            "soc_error.sigma_zero_reproduces_deterministic",
            "bitwise equal",
            is_true,
            boolean,
        ),
    ]
    return claims


def verdict_is(expected: str) -> Callable[[object], bool]:
    return lambda value: value == expected


def text(value: object) -> str:
    return str(value)


def soc_estimation_claims() -> list[CheckedClaim]:
    claims = []
    scope = "RMSE, Test A and Test B, margin above the GRU seed spread on both"
    for scenario, label in (
        ("nominal_inputs", "coulomb counting with a 0.5 % FS current offset"),
        ("nominal_inputs_cc_at_tester_accuracy", "coulomb counting with a 0.1 % FS current offset"),
    ):
        base = f"soc_estimation.comparisons.{scenario}"
        claims += [
            CheckedClaim(
                f"The GRU beats the EKF ({scope})",
                f"{base}.gru_vs_ekf.verdict",
                "gru_beats_ekf",
                verdict_is("gru_beats_ekf"),
                text,
                "EKF on a 1RC model; GRU trained at 0, 25, 40 C",
            )
            if scenario == "nominal_inputs"
            else None,
            CheckedClaim(
                f"The GRU beats {label} ({scope})",
                f"{base}.gru_vs_cc.verdict",
                "gru_beats_cc",
                verdict_is("gru_beats_cc"),
                text,
            ),
            CheckedClaim(
                f"The EKF beats {label} ({scope})",
                f"{base}.ekf_vs_cc.verdict",
                "ekf_beats_cc",
                verdict_is("ekf_beats_cc"),
                text,
            ),
        ]
    claims = [c for c in claims if c is not None]
    for estimator, name in (("ekf", "EKF"), ("ekf2", "2RC EKF")):
        for test_set in ("test_a", "test_b"):
            claims.append(
                CheckedClaim(
                    f"{name} error 300 s after a 20 % wrong start ({test_set}, mean absolute)",
                    f"soc_estimation.results.{estimator}.wrong_start.{test_set}.all."
                    "abs_error_at_300s_percent_mean",
                    "<= 2 % SOC",
                    at_most(2.0),
                    lambda v: f"{float(v):.3g} % SOC",
                )
            )
    claims += [
        CheckedClaim(
            f"The 2RC EKF with hysteresis beats the {other_label} ({scope})",
            f"soc_estimation.comparisons.nominal_inputs.{pair}.verdict",
            expected,
            verdict_is(expected),
            text,
            "same fixed splits; EKF parameters tuned on validation files only",
        )
        for pair, other_label, expected in (
            ("ekf2_vs_ekf", "1RC EKF", "ekf2_beats_ekf"),
            ("ekf2_vs_gru", "GRU", "ekf2_beats_gru"),
        )
    ]
    for technique in ("passive", "buck_boost"):
        for estimator in ("ekf", "ekf2", "gru"):
            claims.append(
                CheckedClaim(
                    f"Balancing on {estimator.upper()} SOC ({technique}) brings most packs "
                    "to the 1 % true spread target",
                    f"soc_estimation.feedthrough.techniques.{technique}.{estimator}."
                    "fraction_true_spread_above_tolerance",
                    "<= 50 % of packs above target",
                    at_most(0.5),
                    percent,
                    "bias drawn from the measured error pool, 200 packs",
                )
            )
    return claims


def scaling_claims() -> list[CheckedClaim]:
    claims = []
    for profile, text_profile in (
        ("step", "step initial profile"),
        ("random", "random initial SOCs"),
    ):
        for key, label in (
            ("passive", "passive shunt"),
            ("buck_boost", "inductor buck-boost"),
            ("flying_capacitor", "flying capacitor"),
        ):
            if profile == "random" and key == "flying_capacitor":
                continue
            claims.append(
                CheckedClaim(
                    f"{label}: balancing time vs N follows the derived power law ({text_profile})",
                    f"scaling.profiles.{profile}.{key}.agrees_with_theory",
                    "|fitted - theory slope| <= 0.25",
                    is_true,
                    boolean,
                    "log-log fit over N >= 8",
                )
            )
    for key, label in (
        ("passive", "passive"),
        ("buck_boost", "buck-boost"),
        ("flying_capacitor", "flying capacitor"),
    ):
        claims.append(
            CheckedClaim(
                f"Phase 3 physics baseline for {label} still predicts balancing time at N = 32 "
                "(random initial SOCs, median error)",
                f"scaling.profiles.random.{key}.physics_time_median_abs_pct_error_by_n.32",
                "<= 10 %",
                at_most(10.0),
                lambda v: f"{float(v):.3g} %",
            )
        )
    return claims


def rl_claims() -> list[CheckedClaim]:
    claims = []
    for key, label in (
        ("none", "no balancing"),
        ("rule_based", "the midsem rule-based controller"),
        ("greedy", "greedy maximum-transfer balancing"),
    ):
        claims.append(
            CheckedClaim(
                f"The PPO policy beats {label} on the reward (Ah minus lambda times loss), "
                "mean paired gain above two standard errors on every test cycle",
                f"rl_control.verdicts.{key}.reward.rl_beats",
                "all 4 test cycles, > 2 SE",
                is_true,
                boolean,
                "100 fresh packs per cycle, 3 seeds averaged; null result is reported as such",
            )
        )
    claims.append(
        CheckedClaim(
            "No cell leaves its voltage window before the episode ends, for every controller "
            "and test pack (seconds above 4.2 V, summed over 100 packs)",
            "rl_control.limit_violation_seconds_total",
            "= 0 s",
            at_most(0.0),
            lambda v: f"{float(v):.3g} s",
        )
    )
    return claims


def midsem_claims() -> list[CheckedClaim]:
    claims = [
        CheckedClaim(
            f"The midsem baseline for {label} (time, energy, efficiency) is reproduced by the "
            "assumed 3 Ah cell model",
            f"midsem_consistency.baseline.{key}.max_relative_difference_reproduced_vs_midsem",
            "<= 0.1 %",
            at_most(0.001),
            percent,
            "generic NMC cell of the midsem, not the fitted HG2 cell",
        )
        for key, label in (
            ("passive", "passive"),
            ("buck_boost", "buck-boost"),
            ("flying_capacitor", "flying capacitor"),
        )
    ]
    for target in ("balancing_time_s", "energy_dissipated_j"):
        claims.append(
            CheckedClaim(
                f"Retraining the surrogate on the assumed model reproduces the midsem test R2 "
                f"for {target}",
                f"midsem_consistency.surrogate_retrained_on_assumed_model.{target}.r2",
                "within 1e-4 of the midsem value",
                lambda v, t=target: abs(float(v) - MIDSEM_R2[t]) <= 1e-4,
                lambda v: f"{float(v):.4f}",
            )
        )
    return claims


MIDSEM_R2 = {"balancing_time_s": 0.9726, "energy_dissipated_j": 0.9946}


def converter_audit_claims() -> list[CheckedClaim]:
    base = "converter_audit"
    return [
        CheckedClaim(
            "Buck-boost cell R0 loss: the RMS (triangular) current model raises the baseline "
            "dissipated energy compared with the average-current default (flag only, default "
            "unchanged)",
            f"{base}.buck_boost.ripple_r0.baseline.energy_ratio",
            "reported; ratio > 1",
            lambda v: float(v) > 1.0,
            lambda v: f"{float(v):.4f}",
            "default stays on average current until circuit validation",
        ),
        CheckedClaim(
            "Energy ledger still closes with the RMS current flag on (100 random packs, "
            "relative to removed energy)",
            f"{base}.buck_boost.ripple_r0.dataset_range.ledger_gap_vs_removed_max_rms_model",
            "<= 1 %",
            at_most(0.01),
            percent,
        ),
        CheckedClaim(
            "Buck-boost link stays in discontinuous conduction for every cell voltage pair "
            "from 3.0 to 4.2 V (D + D2 below 1)",
            f"{base}.buck_boost.peak_current.dcm.discontinuous_everywhere",
            "D + D2 < 1",
            is_true,
            boolean,
            "margin 4 % at the corner",
        ),
        CheckedClaim(
            "Worst-case inductor peak current over the L and f range stays below 20 A",
            f"{base}.buck_boost.peak_current.worst_case.peak_current_a",
            "<= 20 A",
            at_most(20.0),
            lambda v: f"{float(v):.3g} A",
            "L = 10 uH, f = 10 kHz, 4.2 V; saturation rating of a real inductor not checked",
        ),
        CheckedClaim(
            "The switched-capacitor form (1/fC) coth(1/(4 f R C)) equals the two-phase "
            "flying-capacitor link formula for equal phase resistances",
            f"{base}.flying_capacitor."
            "coth_equals_two_phase_for_equal_phase_resistance_max_relative_difference",
            "<= 1e-9",
            at_most(1e-9),
            scientific,
            "the default model has used this form since Phase 2",
        ),
    ]


def voltage_claims() -> list[CheckedClaim]:
    claims = []
    for key, label in (("passive", "passive shunt"), ("buck_boost", "inductor buck-boost")):
        for sigma in (1, 2, 5, 10):
            claims.append(
                CheckedClaim(
                    f"Rest-voltage balancing with {sigma} mV cell sensor error and a 0.5 % "
                    f"stopping threshold brings at least 95 % of {label} packs below 1 % true "
                    "spread (200 packs)",
                    f"voltage_balancing.results.{key}.stop_0p5pct_sigma_{sigma}mv."
                    "fraction_true_spread_below_target",
                    ">= 95 % of packs",
                    lambda v: float(v) >= 0.95,
                    percent,
                    "sensor model: fixed offset plus 60 s averaged noise, rest voltage mapped by "
                    "the fitted OCV curve",
                )
            )
    claims.append(
        CheckedClaim(
            "Rest-voltage balancing stopping exactly at the 1 % target (no margin) brings at "
            "least 95 % of passive packs below 1 % true spread at 0.5 mV sensor error",
            "voltage_balancing.results.passive.stop_1pct_sigma_0p5mv."
            "fraction_true_spread_below_target",
            ">= 95 % of packs",
            lambda v: float(v) >= 0.95,
            percent,
            "a stopping threshold below the target is needed",
        )
    )
    return claims


def selector_claims() -> list[CheckedClaim]:
    claims = []
    for hours in ("0.5", "1", "1.5", "2", "3", "4", "6", "8"):
        base = f"technique_selector.by_budget.{hours.replace('.', 'p')}.policies.selector"
        claims += [
            CheckedClaim(
                f"Surrogate selector picks the oracle technique for a {hours} h parking budget "
                "(300 held-out packs)",
                f"{base}.agrees_with_oracle",
                ">= 90 %",
                lambda v: float(v) >= 0.9,
                percent,
            ),
            CheckedClaim(
                f"Surrogate selector misses a {hours} h budget on at most 5 % of the packs "
                "where some technique meets it",
                f"{base}.budget_violation_fraction_among_feasible",
                "<= 5 %",
                at_most(0.05),
                percent,
            ),
        ]
    return claims


def coupling_claims() -> list[CheckedClaim]:
    return [
        CheckedClaim(
            "Net-current link models (shared cell resistance solved) close the net-current "
            "ledger for buck-boost and flying capacitor (200 random packs worst case, "
            "relative to dissipated energy)",
            "link_coupling.max_net_ledger_gap_vs_dissipated",
            "<= 1 %",
            at_most(0.01),
            percent,
            "per-link model remains the default for continuity with midsem results",
        )
    ]


def drive_population_claims() -> list[CheckedClaim]:
    claims = []
    for key, label in (
        ("buck_boost", "inductor buck-boost"),
        ("flying_capacitor", "flying capacitor"),
    ):
        for scenario in ("driving", "charging"):
            claims.append(
                CheckedClaim(
                    f"{label} raises usable pack Ah while {scenario} for most random packs "
                    "(measured Mixed 1 current, 200 packs)",
                    f"drive_cycle_population.{scenario}.{key}.fraction_packs_with_gain",
                    ">= 90 % of packs",
                    lambda v: float(v) >= 0.9,
                    percent,
                )
            )
    return claims


def checked_claims() -> list[CheckedClaim]:
    claims = [
        CheckedClaim(
            "Under the ASSUMED parameter set the code reproduces the midsem baseline "
            "times, energies and efficiencies at their printed precision",
            "physics_checks.midsem_regression.pass",
            "all 9 values equal",
            is_true,
            boolean,
        ),
        CheckedClaim(
            "Passive balancing time follows the closed form t = max_i 3600 Q_i (R_b + R0_i + R1) "
            "integral of dSOC/OCV from SOC_min + 1 % to SOC_i (baseline and 200 random packs)",
            "physics_checks.passive_closed_form.max_relative_error",
            "<= 3 %",
            at_most(0.03),
            percent,
            "baseline: see physics_checks.passive_closed_form.baseline_relative_error",
        ),
        CheckedClaim(
            "Flying-capacitor balancing conserves charge (Ah out of source cells equals Ah "
            "into sink cells, 200 random packs)",
            "physics_checks.flying_capacitor_charge.max_relative_imbalance",
            "<= 1e-9 relative",
            at_most(1e-9),
            scientific,
        ),
    ]
    claims.extend(validation_claims())
    claims.extend(converter_claims())
    claims.extend(ml_eval_claims())
    for key, label in TECHNIQUES:
        base = f"physics_checks.energy_ledger.{key}"
        claims.append(
            CheckedClaim(
                f"Energy ledger closes for {label}: abs((E_removed - E_delivered) - E_ledger) "
                "/ E_removed over 200 random packs, net cell currents",
                f"{base}.max_net_vs_removed",
                "<= 1 % of E_removed",
                at_most(0.01),
                percent,
            )
        )
    for key, label in TECHNIQUES:
        base = f"physics_checks.energy_ledger.{key}"
        claims.append(
            CheckedClaim(
                f"Reported dissipated energy for {label} agrees with the net-current loss "
                "ledger to within 1 % of the dissipated energy (stricter diagnostic)",
                f"{base}.max_net_vs_dissipated",
                "<= 1 % of E_dissipated",
                at_most(0.01),
                percent,
                LINK_NOTE if key != "passive" else "bleed current is the net cell current",
            )
        )
    for key, label in TECHNIQUES:
        base = f"physics_checks.energy_ledger.{key}"
        claims.append(
            CheckedClaim(
                f"Model-internal consistency for {label}: dissipated energy equals the ledger "
                "with R0 booked per link as the averaged model does",
                f"{base}.max_per_link_vs_dissipated",
                "<= 0.1 % of E_dissipated",
                at_most(0.001),
                percent,
                "residual is first order in dt (halves when dt halves)",
            )
        )
    return (
        claims
        + drive_cycle_claims()
        + temperature_claims()
        + soc_estimation_claims()
        + scaling_claims()
        + rl_claims()
        + coupling_claims()
        + midsem_claims()
        + converter_audit_claims()
        + voltage_claims()
        + selector_claims()
        + drive_population_claims()
    )


UNSUPPORTED = (
    UnsupportedClaim(
        "Learned (RL) control improves balancing under load",
        "PPO reached no gain over no balancing and lost to the rule-based controller on all "
        "four test cycles in two training configurations; a null result",
    ),
    UnsupportedClaim(
        "The GRU or EKF numbers describe other cells, other chemistries or aged cells",
        "one cell, labels from integrated tester current, Mixed 3 absent at 0 and 10 C, no "
        "validation data at 40 C",
    ),
    UnsupportedClaim(
        "The 4-cell ML surrogate predicts strings with N != 4",
        "its input layout is fixed at 4 cells; only the physics baselines were extended to N",
    ),
    UnsupportedClaim(
        "Balancing time scales as N^2 for flying-capacitor strings of random cells",
        "the N^2 slowest-mode law holds for smooth step profiles; random profiles scale slower",
    ),
    UnsupportedClaim(
        "Flying-capacitor balancing is robust to SOC estimation error in general",
        "in this model the link current follows the true voltage difference while the controller "
        "stops on the estimate, so 78 % to 98 % of packs never reach the estimated criterion in "
        "12 h; the result is a property of the model, not a hardware measurement",
    ),
    UnsupportedClaim(
        "Balancing results at 0, 10 or 40 C",
        "only the cell parameters were fitted per temperature; no temperature-dependent "
        "balancing study and the simulator stays at 25 C",
    ),
    UnsupportedClaim(
        "Flying-capacitor link resistance and balancing time match hardware (coth form "
        "against the series form: baseline time 1.96 h against 1.18 h)",
        "only ideal-switch SPICE checks exist; circuit validation with real switches, ESR and "
        "dead time is still open",
    ),
    UnsupportedClaim(
        "The RMS-current R0 loss of the buck-boost (+7 to 8 % energy) is the correct default",
        "an input capacitor on the cell side would smooth the cell current and remove part "
        "of the ripple loss; needs circuit validation",
    ),
    UnsupportedClaim(
        "Drive-cycle balancing gains hold for other packs, spreads, cycles or temperatures",
        "one baseline pack and measured Mixed cycles of one cell at 25 C; no statistics over packs",
    ),
    UnsupportedClaim(
        "The fitted parameters describe other HG2 cells, cell lots or aged cells",
        "one new cell measured at 25 C; no repeat cells or ageing data",
    ),
    UnsupportedClaim(
        "The HWFET cycle validates the fitted model",
        "the 25 C HWFET file in the dataset contains zero current for its full 600 s",
    ),
    UnsupportedClaim(
        "The capacity used for SOC (C/20 discharge, 2.781 Ah) is accurate to better than 1 %",
        "C/20 charge throughput is 2.968 Ah and the two 1C discharges differ by 1.2 %, so "
        "current-measurement offset or scale error cannot be excluded",
    ),
    UnsupportedClaim(
        "Parameter intervals in metrics.json include instrument accuracy and systematic error",
        "intervals come from a residual bootstrap of the HPPC fit and the OCV slope only",
    ),
    UnsupportedClaim(
        "Cell-to-cell variation (Q +/-5 %, R0 +/-10 %) reflects real packs",
        "ASSUMED; the dataset has a single cell, so this stays assumed",
    ),
    UnsupportedClaim(
        "The averaged converter models hold outside the six checked points",
        "buck-boost checked only in discontinuous conduction at D = 0.4 and three (L, f) points; "
        "flying capacitor only at three (C, f) points and 4-switch two-phase operation",
    ),
    UnsupportedClaim(
        "Converter efficiencies and losses match a built converter",
        "SPICE used ideal DC sources and ideal-threshold switches; switching transition, gate "
        "drive, dead-time and core losses are not modelled, so efficiency is an upper bound",
    ),
    UnsupportedClaim(
        "The flying-capacitor split between cell R0 heating and converter heating is exact",
        "R0 loss is booked with the mean link current; SPICE shows pulsed currents, so the element "
        "loss differs by 23 to 43 % (total dissipated power agrees)",
    ),
    UnsupportedClaim(
        "Converter component values (switch resistance, ESR, inductor resistance, duty) are "
        "measured",
        "ASSUMED; no hardware data, only the vendor range checks in the README",
    ),
    UnsupportedClaim(
        "The vendor datasheets validate the model",
        "range checks only: different topologies; the LTC3300-1 efficiency curve values were "
        "not read; the BQ76952 application note was not downloaded",
    ),
    UnsupportedClaim(
        "The reported energy dissipated by the active schemes is physically exact",
        "per-link R0 booking differs from the net-current ledger (FAIL rows above); "
        "a coupled link solution has not been implemented",
    ),
    UnsupportedClaim(
        "ML speedups over simulation are fixed factors",
        "measured in results/timing.json for batch sizes 1 and 1000 on one host CPU; they depend "
        "on the machine and batch size, and the physics formulas are faster than the HGB model",
    ),
    UnsupportedClaim(
        "The plain HGB model is reliable outside its training range",
        "median error rises several-fold when it is tested above the 80th percentile of the "
        "component value or above 8 % initial spread (ml_eval.out_of_range); only the hybrid "
        "and the physics-based models hold up there",
    ),
    UnsupportedClaim(
        "ML results or prediction intervals carry over to other cells or to strings longer than "
        "4 cells",
        "the dataset is simulated from one fitted cell and one pack layout; intervals are "
        "calibrated on that distribution only",
    ),
    UnsupportedClaim(
        "Balancing results hold when SOC is estimated rather than known exactly",
        "SOC is assumed known exactly (Phases 4 and 8)",
    ),
    UnsupportedClaim(
        "The ML model fits and runs on a microcontroller",
        "no embedded export or target measurement (Phase 5)",
    ),
    UnsupportedClaim(
        "Balancing results hold under driving or charging current",
        "only rest balancing is simulated (Phase 6)",
    ),
    UnsupportedClaim(
        "Balancing results hold at temperatures other than the parameter-set temperature",
        "no temperature dependence (Phase 7)",
    ),
    UnsupportedClaim(
        "Results extend to strings longer than 4 cells",
        "only 4-cell packs simulated (Phase 9)",
    ),
)


def lookup(metrics: dict, dotted: str) -> object | None:
    node: object = metrics
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def checked_row(claim: CheckedClaim, metrics: dict) -> str:
    value = lookup(metrics, claim.evidence)
    if value is None:
        return row(claim.text, claim.evidence, "missing", claim.threshold, "NOT RUN", claim.note)
    status = "PASS" if claim.passes(value) else "FAIL"
    return row(claim.text, claim.evidence, claim.render(value), claim.threshold, status, claim.note)


def row(*cells: str) -> str:
    return "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |"


def render(metrics: dict) -> str:
    lines = [
        "# Claims",
        "",
        "Generated by `scripts/report_claims.py` from `results/metrics.json`. Do not edit by hand.",
        f"Parameter set used by run_all.py: `{metrics.get('parameter_set', 'unknown')}`.",
        "",
        *HEADER,
        *(checked_row(claim, metrics) for claim in checked_claims()),
        *(row(claim.text, "-", "-", "-", "NOT SUPPORTED", claim.reason) for claim in UNSUPPORTED),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    CLAIMS_PATH.write_text(render(metrics), newline="\n")
    print(f"wrote {CLAIMS_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
