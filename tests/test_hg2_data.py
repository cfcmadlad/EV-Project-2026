import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.hg2 import TestKind, classify, discharge_sign, load_recording
from src.identification import usable_drive_cycles

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "hg2"
PROVENANCE = json.loads((ROOT / "data" / "processed" / "provenance.json").read_text())
needs_raw = pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw/hg2 not downloaded")


def test_provenance_records_a_verified_download_of_every_file() -> None:
    assert PROVENANCE["dataset"]["doi"] == "10.17632/b5mj79w5w9.2"
    assert PROVENANCE["dataset"]["licence"] == "CC BY 4.0"
    for record in PROVENANCE["files"]:
        assert record["http_status"] == 200
        assert record["sha256"] == record["listing_sha256"]
        assert len(record["sha256"]) == 64


def test_provenance_contains_no_email_address() -> None:
    assert "@" not in (ROOT / "data" / "processed" / "provenance.json").read_text()


def test_classify_recognises_descriptors() -> None:
    assert classify("551_UDDS.csv") == (551, TestKind.UDDS)
    assert classify("549_Dis_0p5C.csv") == (549, TestKind.DISCHARGE_HALF_C)
    assert classify("551_Mixed1.csv") == (551, TestKind.MIXED)
    with pytest.raises(ValueError):
        classify("551_Foo.csv")


@needs_raw
def test_raw_files_match_recorded_hashes() -> None:
    for record in PROVENANCE["files"]:
        flat = record["folder"] in ("root", "25degC")
        path = RAW_DIR / record["name"] if flat else RAW_DIR / record["folder"] / record["name"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]


@needs_raw
def test_discharge_current_is_negative_in_the_cycler_files() -> None:
    assert discharge_sign(load_recording(RAW_DIR / "549_C20DisCh.csv")) == -1.0


@needs_raw
def test_hwfet_file_has_no_current() -> None:
    recordings = {
        kind: [load_recording(RAW_DIR / f"551_{kind.value}.csv")]
        for kind in (TestKind.UDDS, TestKind.HWFET, TestKind.LA92, TestKind.US06)
    }
    usable = usable_drive_cycles(recordings)
    assert usable == {
        TestKind.UDDS: True,
        TestKind.HWFET: False,
        TestKind.LA92: True,
        TestKind.US06: True,
    }
    assert np.all(recordings[TestKind.HWFET][0].current_a == 0.0)
