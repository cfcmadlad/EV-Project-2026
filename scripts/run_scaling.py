import json
import sys
from time import perf_counter

from joblib import Parallel, delayed

from src.config import CONFIG, Technique
from src.experiments import random_streams
from src.scaling import (
    CACHE_PATH,
    PROFILES,
    cache_inputs,
    run_string,
    sample_string_packs,
    step_profile,
    summarise_arrays,
)


def main() -> int:
    spec = CONFIG.scaling
    rngs = random_streams(CONFIG.seed)["scaling"].spawn(len(spec.cells))
    base = {n: sample_string_packs(CONFIG, n, rng) for n, rng in zip(spec.cells, rngs, strict=True)}
    cases = {"random": base, "step": {n: step_profile(p) for n, p in base.items()}}
    jobs = sorted(
        ((profile, n, t) for profile in PROFILES for n in spec.cells for t in Technique),
        key=lambda job: -job[1],
    )
    start = perf_counter()
    outputs = Parallel(n_jobs=-1)(
        delayed(run_string)(CONFIG, n, cases[profile][n], t) for profile, n, t in jobs
    )
    lookup = dict(zip(jobs, outputs, strict=True))
    results = {
        profile: {
            t.key: {str(n): summarise_arrays(lookup[(profile, n, t)]) for n in spec.cells}
            for t in Technique
        }
        for profile in PROFILES
    }
    payload = {"inputs": json.loads(json.dumps(cache_inputs(CONFIG))), "results": results}
    CACHE_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {CACHE_PATH.name} in {perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
