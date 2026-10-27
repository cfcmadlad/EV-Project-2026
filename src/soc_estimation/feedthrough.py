import numpy as np

from src.config import FloatArray, ProjectConfig
from src.pack import sample_packs
from src.soc_error import run_bias_study, summarise_study

ZERO_LABEL = "zero"


def pool_summary(pool: FloatArray) -> dict[str, float]:
    return {
        "n": int(pool.size),
        "mean": float(pool.mean()),
        "std": float(pool.std()),
        "p05": float(np.percentile(pool, 5.0)),
        "p95": float(np.percentile(pool, 95.0)),
    }


def feedthrough_block(
    config: ProjectConfig, pools: dict[str, FloatArray], rng: np.random.Generator
) -> dict[str, object]:
    packs = sample_packs(config.soc_estimation.feedthrough_packs, rng, config.cell, config.pack)
    shape = packs.soc0.shape
    biases = {ZERO_LABEL: np.zeros(shape)}
    biases |= {name: rng.choice(pool, size=shape) for name, pool in pools.items()}
    study = run_bias_study(packs, biases, config)
    return {
        "n_packs": packs.n_packs,
        "pools": {name: pool_summary(pool) for name, pool in pools.items()},
        "techniques": summarise_study(study, ZERO_LABEL, config),
    }
