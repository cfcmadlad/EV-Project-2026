import json
import sys
from time import perf_counter

from joblib import Parallel, delayed

from src.config import CONFIG
from src.drive_cycle import load_profiles
from src.experiments import random_streams
from src.headroom import (
    CACHE_PATH,
    CELLS,
    CYCLES,
    PACKS,
    cache_inputs,
    draw_packs,
    estimation_sigma,
    run_cell,
    summarise,
)


def main() -> int:
    profiles = load_profiles(CONFIG.drive_cycle)
    sigma = estimation_sigma()
    children = random_streams(CONFIG.seed)["headroom"].spawn(2 * len(CELLS))
    packs = {
        n: draw_packs(CONFIG, n, rng) for n, rng in zip(CELLS, children[: len(CELLS)], strict=True)
    }
    bias = {
        n: rng.normal(0.0, sigma, packs[n].soc0.shape)
        for n, rng in zip(CELLS, children[len(CELLS) :], strict=True)
    }
    jobs = sorted(((n, c) for n in CELLS for c in CYCLES), key=lambda job: -job[0])
    start = perf_counter()
    outputs = Parallel(n_jobs=-1)(
        delayed(run_cell)(CONFIG, profiles, n, c, packs[n], bias[n]) for n, c in jobs
    )
    raw = dict(zip(jobs, outputs, strict=True))
    payload = {
        "inputs": json.loads(json.dumps(cache_inputs(CONFIG))),
        "estimation_sigma": sigma,
        "packs_per_cell_count": PACKS,
        "results": summarise(raw),
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {CACHE_PATH.name} in {perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
