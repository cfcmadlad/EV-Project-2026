import json

import pytest

import src.deep_report as deep_report
import src.headroom as headroom
from src.config import CONFIG
from src.fingerprint import stale_paths, text_sha256


def test_stale_paths_finds_nested_blocks():
    metrics = {
        "a": {"status": "ok"},
        "b": {"status": "stale_cache"},
        "c": {"d": {"status": "stale_cache", "x": 1}, "e": {"status": "not_run"}},
    }
    assert stale_paths(metrics) == ["b", "c.d"]


def test_text_hash_ignores_windows_line_endings(tmp_path):
    unix, windows = tmp_path / "u.txt", tmp_path / "w.txt"
    unix.write_bytes(b"a\nb\n")
    windows.write_bytes(b"a\r\nb\r\n")
    assert text_sha256(unix) == text_sha256(windows)


@pytest.mark.parametrize("name", ["soc", "rl", "surrogates", "trajectory"])
def test_deep_caches_with_wrong_inputs_are_marked_stale(monkeypatch, name):
    def loaded(requested):
        if requested == name:
            return {"inputs": {"wrong": 1}, "dataset_sha256": "x", "maps_sha256": "x", "models": {}}
        return None

    monkeypatch.setattr(deep_report, "_load", loaded)
    block = deep_report.deep_models_block("expected")
    assert block[name]["status"] == "stale_cache"


def test_headroom_cache_with_wrong_inputs_raises(monkeypatch, tmp_path):
    path = tmp_path / "headroom_results.json"
    path.write_text(json.dumps({"inputs": {"wrong": 1}, "results": {}}))
    monkeypatch.setattr(headroom, "CACHE_PATH", path)
    with pytest.raises(RuntimeError, match="stale"):
        headroom.load_cache(CONFIG)


def test_headroom_cache_missing_is_not_run(monkeypatch, tmp_path):
    monkeypatch.setattr(headroom, "CACHE_PATH", tmp_path / "absent.json")
    assert headroom.load_cache(CONFIG)["status"] == "not_run"
