from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckedClaim:
    text: str
    evidence: str
    threshold: str
    passes: Callable[[object], bool]
    render: Callable[[object], str]
    note: str = ""


def percent(value: object) -> str:
    return f"{float(value) * 100.0:.4g} %"


def boolean(value: object) -> str:
    return str(bool(value)).lower()


def at_most(limit: float) -> Callable[[object], bool]:
    return lambda value: float(value) <= limit


def is_true(value: object) -> bool:
    return value is True
