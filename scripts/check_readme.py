import json
import sys
from collections.abc import Callable

from src.config import PROJECT_ROOT

Metrics = dict
Check = tuple[str, Callable[[Metrics], str]]


def get(metrics: Metrics, path: str):
    node = metrics
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def f(path: str, spec: str, scale: float = 1.0) -> Callable[[Metrics], str]:
    return lambda m: format(get(m, path) * scale, spec)


CHECKS: list[Check] = [
    ("baseline passive time", f("baseline.0.balancing_time_h", ".2f")),
    ("baseline buck-boost time", f("baseline.1.balancing_time_h", ".2f")),
    ("baseline flying capacitor time", f("baseline.2.balancing_time_h", ".2f")),
    ("driving usable Ah none", f("drive_cycle.scenarios.driving.none.usable_ah", ".3f")),
    (
        "driving usable Ah buck-boost",
        f("drive_cycle.scenarios.driving.buck_boost.usable_ah", ".3f"),
    ),
    (
        "driving usable Ah flying capacitor",
        f("drive_cycle.scenarios.driving.flying_capacitor.usable_ah", ".3f"),
    ),
    (
        "driving gain flying capacitor",
        f("drive_cycle.scenarios.driving.flying_capacitor.gain_vs_none_ah", ".3f"),
    ),
    (
        "driving gain buck-boost",
        f("drive_cycle.scenarios.driving.buck_boost.gain_vs_none_ah", ".3f"),
    ),
    (
        "reference driving",
        f("drive_cycle.reference_packs.charge_equalised.driving_usable_ah", ".3f"),
    ),
    (
        "population buck-boost median gain",
        f("drive_cycle_population.driving.buck_boost.gain_vs_none_ah_median", ".3f"),
    ),
    (
        "population flying capacitor median gain",
        f("drive_cycle_population.driving.flying_capacitor.gain_vs_none_ah_median", ".3f"),
    ),
    (
        "population charging passive",
        f("drive_cycle_population.charging.passive.gain_vs_none_ah_median", ".3f"),
    ),
    (
        "population charging buck-boost",
        f("drive_cycle_population.charging.buck_boost.gain_vs_none_ah_median", ".3f"),
    ),
    (
        "cold model 0 C UDDS whole",
        f("temperature.per_temperature.0.improved_model.validation.UDDS.rmse_whole_v", ".1f", 1000),
    ),
    (
        "cold model 0 C LA92 whole",
        f("temperature.per_temperature.0.improved_model.validation.LA92.rmse_whole_v", ".1f", 1000),
    ),
    (
        "cold model 10 C UDDS window",
        f(
            "temperature.per_temperature.10.improved_model.validation.UDDS.rmse_window_v",
            ".2f",
            1000,
        ),
    ),
    (
        "cold model 25 C UDDS whole",
        f(
            "temperature.per_temperature.25.improved_model.validation.UDDS.rmse_whole_v",
            ".1f",
            1000,
        ),
    ),
    (
        "cold model 40 C UDDS whole",
        f(
            "temperature.per_temperature.40.improved_model.validation.UDDS.rmse_whole_v",
            ".1f",
            1000,
        ),
    ),
    (
        "log-quadratic leave one out",
        f(
            "temperature.r0_forms.log_quadratic_in_inverse_temperature.leave_one_temperature_out_max_relative_error",
            ".1f",
            100,
        ),
    ),
    (
        "arrhenius form residual",
        f("temperature.r0_forms.arrhenius.max_relative_residual", ".1f", 100),
    ),
    ("activation energy", f("temperature.arrhenius.activation_ev", ".3f")),
    (
        "net ledger buck-boost",
        f("link_coupling.buck_boost.net_model_net_ledger_gap_vs_dissipated_max", ".2f", 100),
    ),
    (
        "net ledger flying capacitor",
        f("link_coupling.flying_capacitor.net_model_net_ledger_gap_vs_dissipated_max", ".2f", 100),
    ),
    (
        "per-link ledger flying capacitor",
        f(
            "link_coupling.flying_capacitor.per_link_model_net_ledger_gap_vs_dissipated_max",
            ".1f",
            100,
        ),
    ),
    (
        "2RC EKF test A rmse",
        f("soc_estimation.results.ekf2.nominal.test_a.all.rmse_percent", ".2f"),
    ),
    (
        "2RC EKF test B rmse",
        f("soc_estimation.results.ekf2.nominal.test_b.all.rmse_percent", ".2f"),
    ),
    ("1RC EKF test A rmse", f("soc_estimation.results.ekf.nominal.test_a.all.rmse_percent", ".2f")),
    ("1RC EKF test B rmse", f("soc_estimation.results.ekf.nominal.test_b.all.rmse_percent", ".2f")),
    ("GRU test A rmse", f("soc_estimation.results.gru.nominal.test_a.all.rmse_percent", ".2f")),
    ("GRU test B rmse", f("soc_estimation.results.gru.nominal.test_b.all.rmse_percent", ".2f")),
    (
        "coulomb counting test A",
        f("soc_estimation.results.cc.offset_injected.test_a.all.rmse_percent", ".1f"),
    ),
    (
        "2RC EKF wrong start test B",
        f(
            "soc_estimation.results.ekf2.wrong_start.test_b.all.abs_error_at_300s_percent_mean",
            ".2f",
        ),
    ),
    (
        "feedthrough passive ekf2 median",
        f(
            "soc_estimation.feedthrough.techniques.passive.ekf2.true_final_spread_median",
            ".1f",
            100,
        ),
    ),
    (
        "feedthrough buck-boost gru median",
        f(
            "soc_estimation.feedthrough.techniques.buck_boost.gru.true_final_spread_median",
            ".1f",
            100,
        ),
    ),
    (
        "scaling step flying capacitor slope",
        f("scaling.profiles.step.flying_capacitor.fitted_slope_large_n", ".2f"),
    ),
    (
        "scaling step buck-boost slope",
        f("scaling.profiles.step.buck_boost.fitted_slope_large_n", ".2f"),
    ),
    (
        "scaling random buck-boost slope",
        f("scaling.profiles.random.buck_boost.fitted_slope_large_n", ".2f"),
    ),
    (
        "scaling random flying capacitor slope",
        f("scaling.profiles.random.flying_capacitor.fitted_slope_large_n", ".2f"),
    ),
    (
        "midsem surrogate time R2",
        f("midsem_consistency.surrogate_midsem_reference.balancing_time_s.r2", ".4f"),
    ),
    (
        "HG2 surrogate time R2",
        f("midsem_consistency.surrogate_trained_on_fitted_hg2_model.balancing_time_s.r2", ".4f"),
    ),
    (
        "HG2 surrogate energy MAE",
        f(
            "midsem_consistency.surrogate_trained_on_fitted_hg2_model.energy_dissipated_j.mae",
            ".1f",
        ),
    ),
    (
        "ripple baseline energy",
        f("converter_audit.buck_boost.ripple_r0.baseline.energy_j_rms_current", ".1f"),
    ),
    (
        "ripple energy ratio median",
        lambda m: (
            format(
                (
                    get(m, "converter_audit.buck_boost.ripple_r0.dataset_range.energy_ratio_median")
                    - 1.0
                )
                * 100,
                ".1f",
            )
            + " %"
        ),
    ),
    (
        "worst peak current",
        f("converter_audit.buck_boost.peak_current.worst_case.peak_current_a", ".1f"),
    ),
    ("DCM maximum", f("converter_audit.buck_boost.peak_current.dcm.max_d_plus_d2", ".2f")),
    (
        "coth baseline resistance",
        f(
            "converter_audit.flying_capacitor.baseline.coth.link_resistance_ohm_baseline_link",
            ".0f",
            1000,
        ),
    ),
    (
        "coth time ratio",
        f("converter_audit.flying_capacitor.baseline_time_ratio_coth_over_midsem", ".2f"),
    ),
    (
        "selector 1 h agreement",
        f("technique_selector.by_budget.1.policies.selector.agrees_with_oracle", ".1f", 100),
    ),
    (
        "selector 1 h violation",
        f(
            "technique_selector.by_budget.1.policies.selector.budget_violation_fraction_among_feasible",
            ".1f",
            100,
        ),
    ),
    (
        "selector 1 h energy",
        f("technique_selector.by_budget.1.policies.selector.mean_energy_j", ".1f"),
    ),
    (
        "selector 0.5 h energy",
        f("technique_selector.by_budget.0p5.policies.selector.mean_energy_j", ".1f"),
    ),
    (
        "voltage passive 1 mV",
        f(
            "voltage_balancing.results.passive.stop_0p5pct_sigma_1mv.balancing_time_s_median",
            ".0f",
            1 / 3600,
        )
        if False
        else f(
            "voltage_balancing.results.passive.stop_0p5pct_sigma_1mv.energy_dissipated_j_median",
            ".0f",
        ),
    ),
    (
        "voltage required passive",
        f(
            "voltage_balancing.required_voltage_accuracy.passive.stop_0p5pct.interpolated_sigma_mv",
            ".1f",
        ),
    ),
    (
        "voltage required buck-boost",
        f(
            "voltage_balancing.required_voltage_accuracy.buck_boost.stop_0p5pct.interpolated_sigma_mv",
            ".1f",
        ),
    ),
    (
        "voltage required 0.25",
        f(
            "voltage_balancing.required_voltage_accuracy.passive.stop_0p25pct.interpolated_sigma_mv",
            ".1f",
        ),
    ),
    ("rl rule Mixed7", f("rl_control.test.Mixed7.rule_based.pack_ah.mean", ".3f")),
    ("rl none Mixed7", f("rl_control.test.Mixed7.none.pack_ah.mean", ".3f")),
    ("rl ppo UDDS", f("rl_control.test.UDDS.ppo_mean_of_seeds.pack_ah.mean", ".3f")),
    ("rl reference Mixed7", f("rl_control.test.Mixed7.ideal_reference.pack_ah.mean", ".3f")),
]


def main() -> int:
    metrics = json.loads((PROJECT_ROOT / "results" / "metrics.json").read_text())
    readme = (PROJECT_ROOT / "README.md").read_text()
    figures = (PROJECT_ROOT / "results" / "FIGURES.md").read_text()
    text = readme + "\n" + figures
    missing = []
    for label, expected in CHECKS:
        value = expected(metrics)
        if value not in text:
            missing.append(f"{label}: expected '{value}' somewhere in README.md or FIGURES.md")
    for line in missing:
        print(line)
    print(f"{len(CHECKS) - len(missing)} of {len(CHECKS)} numbers found")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
