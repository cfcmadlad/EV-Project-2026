import json
import sys
from time import perf_counter

from src.config import CONFIG
from src.experiments import random_streams
from src.voltage_balancing import CACHE_PATH, run_voltage_study


def main() -> int:
    start = perf_counter()
    payload = run_voltage_study(CONFIG, random_streams(CONFIG.seed)["voltage_balancing"])
    CACHE_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {CACHE_PATH.name} in {perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
