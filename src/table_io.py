import csv
from pathlib import Path


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: format(v, ".9g") if isinstance(v, float) else v for k, v in row.items()}
            )


def read_rows(path: Path) -> list[dict[str, str | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{k: _coerce(v) for k, v in row.items()} for row in rows]


def _coerce(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value
