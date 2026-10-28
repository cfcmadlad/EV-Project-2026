import hashlib
from pathlib import Path

CRLF = bytes((13, 10))
LF = bytes((10,))


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(CRLF, LF)).hexdigest()


def maps_sha256() -> str:
    from src.calibration import circuit_maps

    maps = circuit_maps()
    digest = hashlib.sha256()
    for grid in (
        maps.flying_element_ohm,
        maps.flying_gate_w,
        maps.buck_boost_current_ratio,
        maps.buck_boost_overhead_w,
    ):
        digest.update(",".join(format(v, ".9g") for v in grid.values.ravel()).encode())
    return digest.hexdigest()


def binary_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
