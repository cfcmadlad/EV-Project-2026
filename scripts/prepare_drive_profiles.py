import json
import sys

from src.config import CONFIG, PROJECT_ROOT
from src.hg2 import TestKind, discharge_sign, load_recording
from src.params import SECONDS_PER_HOUR
from src.resample import interval_mean

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hg2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "drive_profiles_25c.json"
PROVENANCE_PATH = PROCESSED_DIR / "provenance.json"
DRIVE_STATUS = "TABLE"
DECIMALS = 6


def main() -> int:
    paths = sorted(RAW_DIR.glob("*_Mixed?.csv"))
    if not paths:
        print(f"no Mixed files in {RAW_DIR}; run scripts/fetch_hg2.py --group mixed25 first")
        return 1
    sign = discharge_sign(load_recording(RAW_DIR / "549_C20DisCh.csv"))
    hashes = {
        record["name"]: record["sha256"]
        for record in json.loads(PROVENANCE_PATH.read_text())["files"]
    }
    step_s = CONFIG.drive_cycle.profile_step_s
    profiles: dict[str, object] = {}
    for path in paths:
        recording = load_recording(path)
        rows = recording.status == DRIVE_STATUS
        current = interval_mean(
            recording.time_s[rows], sign * recording.current_a[rows], step_s
        ).round(DECIMALS)
        name = f"Mixed{path.stem[-1]}"
        profiles[name] = {
            "source_file": path.name,
            "source_sha256": hashes[path.name],
            "duration_s": float(current.size * step_s),
            "discharge_ah": float(current.sum() * step_s / SECONDS_PER_HOUR),
            "peak_discharge_a": float(current.max()),
            "peak_regen_a": float(-current.min()),
            "current_a": current.tolist(),
        }
        assert recording.kind is TestKind.MIXED
    payload = {
        "temperature": "25degC",
        "doi": json.loads(PROVENANCE_PATH.read_text())["dataset"]["doi"],
        "averaging_step_s": step_s,
        "convention": "discharge positive, regeneration negative, amperes",
        "rows_used": f"status {DRIVE_STATUS} (drive part, pause rows excluded)",
        "profiles": profiles,
    }
    OUTPUT_PATH.write_text(json.dumps(payload) + "\n", newline="\n")
    print(f"wrote {len(profiles)} profiles to {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
