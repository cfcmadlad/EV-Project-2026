import json
import sys
from time import perf_counter

import torch
from joblib import Parallel, delayed

from src.config import CONFIG, Technique
from src.drive_cycle import load_profiles
from src.mpc import CACHE_PATH, CYCLES, TECHNIQUES, cache_inputs, draw_packs, run_study, summarise


def job(profiles: dict, n_cells: int, cycle: str, technique: Technique) -> dict[str, object]:
    torch.set_num_threads(1)
    return run_study(CONFIG, profiles, n_cells, cycle, technique, draw_packs(CONFIG, n_cells))


def main() -> int:
    profiles = load_profiles(CONFIG.drive_cycle)
    keys = sorted(
        ((t, n, c) for t in TECHNIQUES for n in CONFIG.mpc.cells for c in CYCLES),
        key=lambda key: -key[1],
    )
    start = perf_counter()
    outputs = Parallel(n_jobs=-1)(delayed(job)(profiles, n, c, t) for t, n, c in keys)
    raw = {(t.key, n, c): out for (t, n, c), out in zip(keys, outputs, strict=True)}
    payload = {
        "inputs": json.loads(json.dumps(cache_inputs(CONFIG))),
        "results": summarise(raw, CONFIG),
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {CACHE_PATH.name} in {perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
