import json
import sys
from dataclasses import replace

from src.config import CONFIG
from src.dataset import generate_dataset
from src.experiments import random_streams
from src.midsem import CACHE_PATH, cache_inputs
from src.params import ASSUMED
from src.predictor import train_and_evaluate


def main() -> int:
    config = replace(CONFIG, params=ASSUMED)
    dataset = generate_dataset(config, random_streams(config.seed)["dataset"])
    report = train_and_evaluate(dataset, config.ml, config.seed)
    payload = {
        "inputs": json.loads(json.dumps(cache_inputs(CONFIG))),
        "ml": {
            name: {"r2": v["overall"]["r2"], "mae": v["overall"]["mae"]}
            for name, v in report.metrics.items()
        },
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(payload["ml"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
