from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_all import run
from src.config import CONFIG, PlotConfig
from src.params import (
    CheckSpec,
    DatasetSpec,
    DriveCycleSpec,
    IdentificationSpec,
    MLEvalSpec,
    MLSpec,
    RlSpec,
    ScalingSpec,
    SocErrorSpec,
    SocEstimationSpec,
    SweepSpec,
)

SMALL_CONFIG = replace(
    CONFIG,
    sweep=SweepSpec(n_points=3),
    dataset=DatasetSpec(runs_per_technique=30),
    ml=MLSpec(max_iter=20, permutation_repeats=2, timing_repeats=1),
    checks=CheckSpec(packs_per_technique=6, dt_scaling_packs=2, dt_scaling_steps_s=(1.0, 0.5)),
    identification=IdentificationSpec(bootstrap_resamples=8),
    plot=PlotConfig(dpi=40),
    scaling=ScalingSpec(cells=(4, 8), packs=4, max_balancing_h=50.0),
    soc_estimation=SocEstimationSpec(feedthrough_packs=4, epochs=1),
    rl=RlSpec(test_packs=4, total_timesteps=2048, seeds=(0,)),
    drive_cycle=DriveCycleSpec(max_cycle_repeats=1, charge_horizon_s=3600),
    soc_error=SocErrorSpec(
        packs=6, max_balancing_h=4.0, deterministic_check_packs=3, sigma_percent=(0.0, 3.0)
    ),
    ml_eval=MLEvalSpec(
        n_folds=3,
        conformal_repeats=2,
        time_grid_points=24,
        bisection_steps=12,
        timing_repeats=1,
        timing_batch_sizes=(1, 30),
    ),
)
WALL_CLOCK_FIELDS = ("time_per_pack", "run_all_s", "physics_checks_s")
COMPARED_FILES = ("metrics.json", "baseline_table.csv", "dataset.csv")


def test_two_runs_give_identical_metrics(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(run, SMALL_CONFIG, folder) for folder in (first, second)]:
            future.result()
    for name in COMPARED_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    metrics = (first / "metrics.json").read_text()
    assert all(field not in metrics for field in WALL_CLOCK_FIELDS)
