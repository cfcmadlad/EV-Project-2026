import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "datasheets"
PROVENANCE_PATH = ROOT / "data" / "processed" / "provenance.json"
VALUES_PATH = ROOT / "data" / "processed" / "datasheet_values.json"
TIMEOUT_S = 30
ATTEMPTS = 2
RETRY_DELAY_S = 5.0
USER_AGENT = "curl/8.0"


@dataclass(frozen=True, slots=True)
class Datasheet:
    key: str
    file: str
    url: str
    required_phrase: str
    present_origin: str


DATASHEETS = (
    Datasheet(
        "bq76952",
        "bq76952.pdf",
        "https://www.ti.com/lit/ds/symlink/bq76952.pdf",
        "BQ76952",
        "downloaded with curl (HTTP 200, application/pdf) before this script re-verified it",
    ),
    Datasheet(
        "ltc3300_1",
        "LTC3300-1.pdf",
        "https://www.analog.com/media/en/technical-documentation/data-sheets/LTC3300-1.pdf",
        "LTC3300-1",
        "saved from the vendor URL through a web browser; direct script downloads from this "
        "host timed out",
    ),
)


class FetchError(Exception):
    pass


def download(item: Datasheet) -> tuple[int, bytes]:
    request = Request(item.url, headers={"Accept": "application/pdf,*/*", "User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with urlopen(request, timeout=TIMEOUT_S) as response:
                return response.status, response.read()
        except (URLError, ConnectionError, TimeoutError) as error:
            last_error = error
            sleep(RETRY_DELAY_S * (attempt + 1))
    raise FetchError(
        f"{item.file}: download failed after {ATTEMPTS} attempts ({last_error}). Save the PDF "
        f"from {item.url} in a browser and place it at data/raw/datasheets/{item.file}, "
        "then run this script again"
    )


def parse_check(item: Datasheet, body: bytes, path: Path) -> str:
    if not body.startswith(b"%PDF"):
        raise FetchError(f"{item.file}: not a PDF")
    reader = PdfReader(path)
    first = reader.pages[0].extract_text() or ""
    if item.required_phrase not in first:
        raise FetchError(f"{item.file}: phrase {item.required_phrase!r} not on page 1")
    return f"pdf, {len(reader.pages)} pages, {item.required_phrase} on page 1"


def obtain(item: Datasheet) -> dict[str, object]:
    path = RAW_DIR / item.file
    status: int | None = None
    if path.exists():
        body = path.read_bytes()
        origin = item.present_origin
    else:
        status, body = download(item)
        if status != 200:
            raise FetchError(f"{item.file}: HTTP {status}")
        path.write_bytes(body)
        origin = "downloaded by scripts/fetch_datasheets.py"
    digest = hashlib.sha256(body).hexdigest()
    expected = json.loads(VALUES_PATH.read_text())[item.key]["sha256"]
    if digest != expected:
        raise FetchError(
            f"{item.file}: sha256 {digest} differs from the transcribed values ({expected}); "
            "the document revision may have changed, so data/processed/datasheet_values.json "
            "must be checked against it"
        )
    return {
        "name": item.file,
        "source_url": item.url,
        "origin": origin,
        "http_status": status,
        "size_bytes": len(body),
        "sha256": digest,
        "parse_check": parse_check(item, body, path),
    }


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for item in DATASHEETS:
        try:
            records.append(obtain(item))
        except FetchError as error:
            failures.append(str(error))
    provenance = json.loads(PROVENANCE_PATH.read_text()) if PROVENANCE_PATH.exists() else {}
    provenance["datasheets"] = records
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", newline="\n")
    for record in records:
        print(f"verified {record['name']}: {record['size_bytes']} bytes, {record['parse_check']}")
    for failure in failures:
        print(f"FETCH FAILED: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
