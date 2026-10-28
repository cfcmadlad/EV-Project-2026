import hashlib
from pathlib import Path

CRLF = bytes((13, 10))
LF = bytes((10,))
MAP_FILES = ("flying_cap.csv", "buck_boost.csv")


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(CRLF, LF)).hexdigest()


def maps_sha256() -> str:
    from src.calibration import LTSPICE_DIR

    digest = hashlib.sha256()
    for name in MAP_FILES:
        digest.update((LTSPICE_DIR / name).read_bytes().replace(CRLF, LF))
    return digest.hexdigest()


def binary_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
