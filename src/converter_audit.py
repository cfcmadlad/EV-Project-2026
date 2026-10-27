from dataclasses import replace

import numpy as np

from src.balancing import simulate
from src.checks import ledger_discrepancy, random_case
from src.config import ProjectConfig, Technique
from src.experiments import baseline_pack
from src.params import SECONDS_PER_HOUR, FlyingCapModel
from src.switched_cap import coth_resistance_ohm, link_resistance_ohm

CELL_VOLTAGE_RANGE_V = (3.0, 4.2)
NOMINAL_CELL_VOLTAGE_V = 3.7
GRID_POINTS = 25


def with_buck_boost(config: ProjectConfig, **changes) -> ProjectConfig:
    spec = replace(config.buck_boost, **changes)
    return replace(config, params=replace(config.params, buck_boost=spec))


def with_flying_model(config: ProjectConfig, model: FlyingCapModel) -> ProjectConfig:
    spec = replace(config.flying_capacitor, model=model)
    return replace(config, params=replace(config.params, flying_capacitor=spec))


def baseline_run(config: ProjectConfig, technique: Technique) -> tuple[float, float]:
    hardware = config.technique(technique)
    result = simulate(
        technique,
        baseline_pack(config),
        np.array([hardware.component.mid]),
        np.array([hardware.frequency_hz.mid]),
        config,
    )
    return float(result.balancing_time_s[0]), float(result.energy_dissipated_j[0])


def ripple_audit(config: ProjectConfig, rng: np.random.Generator) -> dict[str, object]:
    flagged = with_buck_boost(config, ripple_r0_loss=True)
    time_off, energy_off = baseline_run(config, Technique.BUCK_BOOST)
    time_on, energy_on = baseline_run(flagged, Technique.BUCK_BOOST)
    packs, component, frequency = random_case(
        Technique.BUCK_BOOST, config.checks.audit_packs, config, rng
    )
    plain = simulate(Technique.BUCK_BOOST, packs, component, frequency, config)
    ripple = simulate(Technique.BUCK_BOOST, packs, component, frequency, flagged)
    energy_ratio = ripple.energy_dissipated_j / plain.energy_dissipated_j
    duty = config.buck_boost.duty
    return {
        "rms_to_average_squared_ratio_source": 4.0 / (3.0 * duty),
        "baseline": {
            "time_s_average_current": time_off,
            "time_s_rms_current": time_on,
            "energy_j_average_current": energy_off,
            "energy_j_rms_current": energy_on,
            "energy_ratio": energy_on / energy_off,
        },
        "dataset_range": {
            "n_packs": config.checks.audit_packs,
            "energy_ratio_median": float(np.median(energy_ratio)),
            "energy_ratio_p95": float(np.percentile(energy_ratio, 95.0)),
            "energy_ratio_max": float(energy_ratio.max()),
            "time_ratio_median": float(np.median(ripple.balancing_time_s / plain.balancing_time_s)),
            "ledger_gap_vs_removed_max_rms_model": float(
                ledger_discrepancy(ripple)["net_vs_removed"].max()
            ),
        },
        "default_uses_rms_current": bool(config.buck_boost.ripple_r0_loss),
    }


def peak_current_audit(config: ProjectConfig) -> dict[str, object]:
    spec = config.buck_boost
    inductance = np.geomspace(spec.component.low, spec.component.high, GRID_POINTS)
    frequency = np.linspace(spec.frequency_hz.low, spec.frequency_hz.high, GRID_POINTS)
    voltage = np.array([CELL_VOLTAGE_RANGE_V[0], NOMINAL_CELL_VOLTAGE_V, CELL_VOLTAGE_RANGE_V[1]])
    peak = (
        voltage[:, None, None] * spec.duty / (inductance[None, :, None] * frequency[None, None, :])
    )
    index = np.unravel_index(np.argmax(peak), peak.shape)
    low, high = CELL_VOLTAGE_RANGE_V
    ratio = np.linspace(low / high, high / low, GRID_POINTS)
    d_plus_d2 = spec.duty * (1.0 + ratio)
    return {
        "formula": "I_pk = V D / (L f)",
        "duty": spec.duty,
        "worst_case": {
            "peak_current_a": float(peak.max()),
            "cell_voltage_v": float(voltage[index[0]]),
            "inductance_h": float(inductance[index[1]]),
            "frequency_hz": float(frequency[index[2]]),
        },
        "at_nominal_voltage_worst_corner_a": float(peak[1].max()),
        "baseline_midrange_a": float(
            NOMINAL_CELL_VOLTAGE_V * spec.duty / (spec.component.mid * spec.frequency_hz.mid)
        ),
        "dcm": {
            "max_d_plus_d2": float(d_plus_d2.max()),
            "margin_to_boundary": float(1.0 - d_plus_d2.max()),
            "discontinuous_everywhere": bool(d_plus_d2.max() < 1.0),
            "voltage_range_v": list(CELL_VOLTAGE_RANGE_V),
        },
    }


def flying_capacitor_audit(config: ProjectConfig) -> dict[str, object]:
    spec = config.flying_capacitor
    pack = baseline_pack(config)
    r0_mean = float(pack.r0_ohm.mean())
    phase_ohm = r0_mean + 2.0 * spec.r_ds_on_ohm + spec.esr_ohm
    capacitance = np.geomspace(spec.component.low, spec.component.high, GRID_POINTS)
    frequency = np.linspace(spec.frequency_hz.low, spec.frequency_hz.high, GRID_POINTS)
    inverse = 1.0 / (capacitance[:, None] * frequency[None, :])
    forms = {
        "midsem_series": inverse + 2.0 * spec.r_ds_on_ohm + spec.esr_ohm + 2.0 * r0_mean,
        "two_phase": coth_resistance_ohm(inverse, np.full_like(inverse, phase_ohm)),
        "coth": coth_resistance_ohm(inverse, np.full_like(inverse, phase_ohm)),
    }
    mid_inverse = np.array([1.0 / (spec.component.mid * spec.frequency_hz.mid)])
    r0_row = pack.r0_ohm[:1]
    at_baseline = {}
    runs = {}
    for model in FlyingCapModel:
        local = replace(spec, model=model)
        at_baseline[model.value] = float(link_resistance_ohm(mid_inverse, r0_row, local)[0, 0])
        time_s, energy_j = baseline_run(
            with_flying_model(config, model), Technique.FLYING_CAPACITOR
        )
        runs[model.value] = {
            "link_resistance_ohm_baseline_link": at_baseline[model.value],
            "balancing_time_s": time_s,
            "balancing_time_h": time_s / SECONDS_PER_HOUR,
            "energy_dissipated_j": energy_j,
        }
    return {
        "coth_formula": "R_eq = (1/(f C)) coth(1/(4 f R C)), R = r0 + 2 Rds_on + ESR per phase",
        "coth_equals_two_phase_for_equal_phase_resistance_max_relative_difference": float(
            np.max(np.abs(forms["coth"] / forms["two_phase"] - 1.0))
        ),
        "range": {
            name: {"min_ohm": float(v.min()), "max_ohm": float(v.max())}
            for name, v in forms.items()
        },
        "series_over_coth_at_range_extremes": {
            "min": float(np.min(forms["midsem_series"] / forms["coth"])),
            "max": float(np.max(forms["midsem_series"] / forms["coth"])),
        },
        "baseline": runs,
        "baseline_time_ratio_coth_over_midsem": runs["coth"]["balancing_time_s"]
        / runs["midsem"]["balancing_time_s"],
        "baseline_energy_ratio_coth_over_midsem": runs["coth"]["energy_dissipated_j"]
        / runs["midsem"]["energy_dissipated_j"],
        "default_model": config.flying_capacitor.model.value,
    }


def converter_audit_block(config: ProjectConfig, rng: np.random.Generator) -> dict[str, object]:
    return {
        "status": "ok",
        "buck_boost": {
            "ripple_r0": ripple_audit(config, rng),
            "peak_current": peak_current_audit(config),
        },
        "flying_capacitor": flying_capacitor_audit(config),
    }
