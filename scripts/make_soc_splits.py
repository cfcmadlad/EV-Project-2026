import json
import sys

from src.config import CONFIG
from src.soc_estimation.data import (
    CYCLE_PATTERN,
    PROVENANCE_PATH,
    SPLITS_PATH,
    inspect_file,
    mixed_index,
    temperature_capacities,
)

FOLDER = {0: "0degC", 10: "10degC", 25: "25degC", 40: "40degC"}


def main() -> int:
    spec = CONFIG.soc_estimation
    provenance = json.loads(PROVENANCE_PATH.read_text())
    capacities = temperature_capacities()
    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test_a": [],
        "test_b": [],
    }
    rejected: list[dict[str, object]] = []
    absent: list[str] = []
    seen: set[tuple[int, str]] = set()
    for temperature, folder in FOLDER.items():
        for record in provenance["files"]:
            match = CYCLE_PATTERN.match(record["name"])
            if record["folder"] != folder or match is None:
                continue
            cycle = match["cycle"]
            info = inspect_file(folder, record["name"], temperature, capacities[temperature], spec)
            entry = {
                "file": record["name"],
                "folder": folder,
                "temperature_c": temperature,
                "cycle": cycle,
                "sha256": record["sha256"],
                "duration_s": info["duration_s"],
                "discharge_ah": info["discharge_ah"],
            }
            if not info["usable"]:
                rejected.append({**entry, "reason": info["reason"]})
                continue
            seen.add((temperature, cycle))
            index = mixed_index(cycle)
            validation_mixed = dict(spec.validation_mixed_override).get(
                temperature, spec.validation_mixed
            )
            if temperature == spec.test_b_temperature_c:
                splits["test_b"].append(entry)
            elif (
                temperature in spec.train_temperatures_c
                and index in spec.train_mixed
                and index not in validation_mixed
            ):
                splits["train"].append(entry)
            elif temperature in spec.train_temperatures_c and index in validation_mixed:
                splits["validation"].append(entry)
            elif temperature in spec.train_temperatures_c and cycle in spec.test_a_cycles:
                splits["test_a"].append(entry)
    for temperature in (*spec.train_temperatures_c, spec.test_b_temperature_c):
        wanted = [f"Mixed{i}" for i in (*spec.train_mixed, *spec.validation_mixed)]
        wanted += list(spec.test_a_cycles) + ["LA92"]
        absent += [
            f"{c} at {temperature} C" for c in dict.fromkeys(wanted) if (temperature, c) not in seen
        ]
    payload = {
        "definition": {
            "reference_soc": "1 at the start of each drive file (preceded by a full charge to "
            "4.2 V with 50 mA cut-off) minus integrated tester current over the C/20 discharge "
            "capacity at that temperature",
            "capacity_ah_by_temperature": {str(t): c for t, c in capacities.items()},
            "step_s": spec.step_s,
            "rows_used": "status TABLE (drive part)",
            "label_accuracy_note": "tester current accuracy is 0.1 % of a 75 A full scale "
            "(dataset readme), so the integrated label can drift by up to 75 mA times the cycle "
            "time",
        },
        "splits": splits,
        "rejected_files": rejected,
        "absent_or_unusable": sorted(set(absent)),
    }
    SPLITS_PATH.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    counts = {k: len(v) for k, v in splits.items()}
    print(f"wrote {SPLITS_PATH.name}: {counts}, rejected {len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
