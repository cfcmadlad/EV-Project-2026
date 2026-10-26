from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.cell import OcvCurve, build_ocv_curve
from src.config import MV_PER_V, FloatArray
from src.hg2 import DRIVE_CYCLES, Recording, TestKind, discharge_sign
from src.identify import (
    CapacityReport,
    OcvResult,
    PulseBlock,
    RcFit,
    Reference,
    TwoRcFit,
    bootstrap_pulses,
    build_ocv,
    capacity_report,
    counter_capacity_ah,
    discharge_rows,
    extended_window,
    find_pulse_blocks,
    fit_rc,
    fit_two_rc,
    interval,
    pulse_data,
    reference_from,
    rest_cross_check,
    slope_bootstrap,
    window_summary,
)
from src.params import ASSUMED, CellParams, IdentificationSpec, OCVTable
from src.validate import Branch, Replay, ReplayMetrics, replay_cell, replay_cycle, replay_metrics

MODEL_1RC = "1rc"
MODEL_2RC = "2rc"
SLOPE_HALF_WIDTH = 0.01
ZERO_CURRENT_A = 1e-6
SLOPE_PER_PERCENT = 0.01


@dataclass(frozen=True, slots=True)
class Identified:
    sign: float
    capacity_ah: float
    capacity: CapacityReport
    ocv: OcvResult
    ocv_first_pass: OcvResult
    blocks: list[PulseBlock]
    fits_1rc: list[RcFit]
    fits_2rc: list[TwoRcFit]
    c20: Recording
    hppc: Recording


@dataclass(frozen=True, slots=True)
class Constants:
    r0_ohm: dict[str, float]
    r1_ohm: dict[str, float]
    c1_farad: dict[str, float]
    fast_r_ohm: dict[str, float]
    fast_tau_s: dict[str, float]
    slow_r_ohm: dict[str, float]
    slow_tau_s: dict[str, float]
    lumped_r1_ohm: dict[str, float]
    lumped_tau_s: dict[str, float]
    lumped_c1_farad: dict[str, float]


@dataclass(frozen=True, slots=True)
class CandidateModels:
    one_rc: CellParams
    two_rc_r0_ohm: float
    two_rc_branches: tuple[Branch, ...]
    lumped: CellParams


def lump_branches(branches: tuple[Branch, ...]) -> tuple[float, float, float]:
    r_total = sum(r for r, _ in branches)
    tau = sum(r * t for r, t in branches) / r_total
    return r_total, tau, tau / r_total


def r0_interpolator(fits: list[RcFit]) -> Callable[[FloatArray], FloatArray]:
    soc = np.array([fit.soc for fit in fits])[::-1]
    r0 = np.array([fit.r0_ohm for fit in fits])[::-1]
    return lambda query: np.interp(query, soc, r0)


def identify(recordings: dict[TestKind, list[Recording]], spec: IdentificationSpec) -> Identified:
    c20, hppc = recordings[TestKind.C20][0], recordings[TestKind.HPPC][0]
    sign = discharge_sign(c20)
    capacity_ah = counter_capacity_ah(c20, discharge_rows(c20))
    capacity = capacity_report(c20, recordings[TestKind.CAPACITY_1C], sign)
    blocks = find_pulse_blocks(hppc, sign, spec)
    ocv_first = build_ocv(c20, sign, capacity_ah, np.zeros_like, spec)
    ocv = ocv_first
    fits_1rc: list[RcFit] = []
    for _ in range(2):
        reference = reference_from(ocv)
        fits_1rc = [
            fit_rc(pulse_data(hppc, sign, capacity_ah, block), reference, spec) for block in blocks
        ]
        ocv = build_ocv(c20, sign, capacity_ah, r0_interpolator(fits_1rc), spec)
    reference = reference_from(ocv)
    fits_1rc = [
        fit_rc(pulse_data(hppc, sign, capacity_ah, block), reference, spec) for block in blocks
    ]
    fits_2rc = [
        fit_two_rc(
            pulse_data(hppc, sign, capacity_ah, extended_window(hppc, block, spec)),
            reference,
            spec,
            fit.r0_ohm,
        )
        for block, fit in zip(blocks, fits_1rc, strict=True)
    ]
    return Identified(
        sign=sign,
        capacity_ah=capacity_ah,
        capacity=capacity,
        ocv=ocv,
        ocv_first_pass=ocv_first,
        blocks=blocks,
        fits_1rc=fits_1rc,
        fits_2rc=fits_2rc,
        c20=c20,
        hppc=hppc,
    )


def constants(found: Identified, spec: IdentificationSpec) -> Constants:
    window = spec.constant_window
    two = found.fits_2rc
    lumped = [lump_branches(((f.r1_ohm, f.tau1_s), (f.r2_ohm, f.tau2_s))) for f in two]
    socs = [f.soc for f in two]

    def over(values: list[float]) -> dict[str, float]:
        selected = np.array(
            [v for v, s in zip(values, socs, strict=True) if window[0] <= s <= window[1]]
        )
        return {
            "median": float(np.median(selected)),
            "min": float(selected.min()),
            "max": float(selected.max()),
            "n_points": int(selected.size),
        }

    return Constants(
        r0_ohm=window_summary(found.fits_1rc, lambda f: f.r0_ohm, window),
        r1_ohm=window_summary(found.fits_1rc, lambda f: f.r1_ohm, window),
        c1_farad=window_summary(found.fits_1rc, lambda f: f.c1_farad, window),
        fast_r_ohm=over([f.r1_ohm for f in two]),
        fast_tau_s=over([f.tau1_s for f in two]),
        slow_r_ohm=over([f.r2_ohm for f in two]),
        slow_tau_s=over([f.tau2_s for f in two]),
        lumped_r1_ohm=over([r for r, _, _ in lumped]),
        lumped_tau_s=over([t for _, t, _ in lumped]),
        lumped_c1_farad=over([c for _, _, c in lumped]),
    )


def candidate_models(found: Identified, values: Constants) -> CandidateModels:
    table = OCVTable(soc=tuple(found.ocv.soc.tolist()), volt_v=tuple(found.ocv.ocv_v.tolist()))
    r0 = values.r0_ohm["median"]
    one_rc = CellParams(
        capacity_ah=found.capacity_ah,
        r0_ohm=r0,
        r1_ohm=values.r1_ohm["median"],
        c1_farad=values.c1_farad["median"],
        ocv=table,
    )
    branches = (
        (values.fast_r_ohm["median"], values.fast_tau_s["median"]),
        (values.slow_r_ohm["median"], values.slow_tau_s["median"]),
    )
    r_total, _, c_total = lump_branches(branches)
    lumped = CellParams(
        capacity_ah=found.capacity_ah, r0_ohm=r0, r1_ohm=r_total, c1_farad=c_total, ocv=table
    )
    return CandidateModels(one_rc=one_rc, two_rc_r0_ohm=r0, two_rc_branches=branches, lumped=lumped)


def usable_drive_cycles(recordings: dict[TestKind, list[Recording]]) -> dict[TestKind, bool]:
    return {
        kind: float(np.abs(recordings[kind][0].current_a).max()) > ZERO_CURRENT_A
        for kind in DRIVE_CYCLES
        if kind in recordings
    }


def replay_all(
    recordings: dict[TestKind, list[Recording]],
    found: Identified,
    models: CandidateModels,
    spec: IdentificationSpec,
) -> tuple[dict[str, dict[str, ReplayMetrics]], dict[str, dict[str, Replay]]]:
    curve: OcvCurve = build_ocv_curve(models.one_rc)
    usable = usable_drive_cycles(recordings)
    metrics: dict[str, dict[str, ReplayMetrics]] = {}
    replays: dict[str, dict[str, Replay]] = {}
    for kind in DRIVE_CYCLES:
        if not usable.get(kind, False):
            continue
        recording = recordings[kind][0]
        runs = {
            "assumed": replay_cell(recording, found.sign, ASSUMED.cell),
            MODEL_1RC: replay_cell(recording, found.sign, models.one_rc),
            MODEL_2RC: replay_cycle(
                recording,
                found.sign,
                found.capacity_ah,
                models.two_rc_r0_ohm,
                models.two_rc_branches,
                curve,
            ),
            "lumped": replay_cell(recording, found.sign, models.lumped),
        }
        replays[kind.value] = runs
        metrics[kind.value] = {name: replay_metrics(run, spec) for name, run in runs.items()}
    return metrics, replays


def passes(metrics: ReplayMetrics, spec: IdentificationSpec) -> bool:
    return (
        metrics.rmse_whole_v <= spec.rmse_whole_cycle_v
        and metrics.rmse_window_v <= spec.rmse_window_v
    )


def check_models(
    metrics: dict[str, dict[str, ReplayMetrics]], spec: IdentificationSpec
) -> dict[str, object]:
    gated = tuple(cycle for cycle in (TestKind.UDDS.value, TestKind.LA92.value) if cycle in metrics)
    verdict = {
        name: all(passes(metrics[cycle][name], spec) for cycle in gated)
        for name in (MODEL_1RC, MODEL_2RC, "lumped", "assumed")
    }
    score = {
        name: float(np.mean([metrics[cycle][name].rmse_whole_v for cycle in gated]))
        for name in (MODEL_1RC, MODEL_2RC)
    }
    if verdict[MODEL_1RC]:
        kept = MODEL_1RC
    elif verdict[MODEL_2RC]:
        kept = MODEL_2RC
    else:
        kept = min(score, key=lambda name: score[name])
    return {
        "gated_cycles": list(gated),
        "pass_by_model": verdict,
        "mean_whole_rmse_v": score,
        "kept_model": kept,
        "kept_model_passes": verdict[kept],
        "status": "PASS" if verdict[kept] else "FAILED",
    }


def table_slope_mv_per_percent(cell: CellParams, soc: float = 0.5) -> float:
    curve = build_ocv_curve(cell)
    low, high = soc - SLOPE_HALF_WIDTH, soc + SLOPE_HALF_WIDTH
    volts = np.interp([low, high], curve.soc, curve.volt)
    return float((volts[1] - volts[0]) / (high - low) * SLOPE_PER_PERCENT * MV_PER_V)


def propagate_inputs(
    found: Identified,
    models: CandidateModels,
    values: Constants,
    spec: IdentificationSpec,
    rng: np.random.Generator,
) -> dict[str, FloatArray]:
    reference = reference_from(found.ocv)
    in_window = [
        i
        for i, fit in enumerate(found.fits_2rc)
        if spec.constant_window[0] <= fit.soc <= spec.constant_window[1]
    ]
    per_block = []
    for i in in_window:
        data = pulse_data(
            found.hppc,
            found.sign,
            found.capacity_ah,
            extended_window(found.hppc, found.blocks[i], spec),
        )
        per_block.append(
            bootstrap_pulses(
                data, reference, found.fits_2rc[i], found.hppc, found.blocks[i], spec, rng
            )
        )
    stacked = np.stack(per_block)
    r0 = np.median(stacked[:, :, 0], axis=0)
    fast_r, fast_tau = np.median(stacked[:, :, 1], axis=0), np.median(stacked[:, :, 2], axis=0)
    slow_r, slow_tau = np.median(stacked[:, :, 3], axis=0), np.median(stacked[:, :, 4], axis=0)
    r_total = fast_r + slow_r
    tau = (fast_r * fast_tau + slow_r * slow_tau) / r_total
    slope = slope_bootstrap(found.ocv, spec, rng)
    tilt = slope["samples"] - slope["slope_v_per_soc"]
    return {
        "r0_ohm": r0,
        "fast_r_ohm": fast_r,
        "fast_tau_s": fast_tau,
        "slow_r_ohm": slow_r,
        "slow_tau_s": slow_tau,
        "r1_ohm": r_total,
        "c1_farad": tau / r_total,
        "ocv_tilt_v_per_soc": tilt,
        "ocv_slope_v_per_soc": slope["samples"],
    }


def interval_table(samples: dict[str, FloatArray], spec: IdentificationSpec) -> dict[str, object]:
    return {name: interval(values, spec.ci_percent) for name, values in samples.items()}


def cross_check(found: Identified) -> dict[str, object]:
    return rest_cross_check(
        found.hppc,
        found.sign,
        found.capacity_ah,
        found.blocks,
        Reference(soc=found.ocv.soc, volt_v=found.ocv.ocv_v),
    )


def fit_tables(found: Identified) -> dict[str, list[float | bool]]:
    one, two = found.fits_1rc, found.fits_2rc
    return {
        "soc": [f.soc for f in one],
        "r0_ohm": [f.r0_ohm for f in one],
        "sample_interval_s": [f.sample_interval_s for f in one],
        "one_rc_r1_ohm": [f.r1_ohm for f in one],
        "one_rc_tau_s": [f.tau_s for f in one],
        "one_rc_c1_farad": [f.c1_farad for f in one],
        "one_rc_rmse_mv": [f.rmse_v * MV_PER_V for f in one],
        "one_rc_at_bound": [f.at_bound for f in one],
        "two_rc_fast_r_ohm": [f.r1_ohm for f in two],
        "two_rc_fast_tau_s": [f.tau1_s for f in two],
        "two_rc_slow_r_ohm": [f.r2_ohm for f in two],
        "two_rc_slow_tau_s": [f.tau2_s for f in two],
        "two_rc_rmse_mv": [f.rmse_v * MV_PER_V for f in two],
        "two_rc_at_bound": [f.at_bound for f in two],
    }
