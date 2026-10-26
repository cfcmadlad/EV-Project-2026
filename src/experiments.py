import numpy as np

from src.balancing import BalanceResult, simulate
from src.config import ProjectConfig, Technique
from src.pack import PackBatch, sample_packs

STREAM_NAMES = ("baseline", "dataset", "checks", "identification", "soc_error")


def random_streams(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.default_rng(seed).spawn(len(STREAM_NAMES))
    return dict(zip(STREAM_NAMES, children, strict=True))


def baseline_pack(config: ProjectConfig) -> PackBatch:
    return sample_packs(
        1,
        random_streams(config.seed)["baseline"],
        config.cell,
        config.pack,
        np.array([config.pack.soc_spread_max]),
    )


def run_baseline(
    pack: PackBatch, config: ProjectConfig, trace: bool = True
) -> dict[Technique, BalanceResult]:
    return {
        technique: simulate(
            technique,
            pack,
            np.array([config.technique(technique).component.mid]),
            np.array([config.technique(technique).frequency_hz.mid]),
            config,
            trace=trace,
        )
        for technique in Technique
    }
