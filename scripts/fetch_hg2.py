import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "hg2"
PROVENANCE_PATH = ROOT / "data" / "processed" / "provenance.json"
API_ROOT = "https://data.mendeley.com/public-api/datasets"
DATASET_ID = "b5mj79w5w9"
DATASET_VERSION = 2
DOI = "10.17632/b5mj79w5w9.2"
LICENCE = "CC BY 4.0"
TITLE = "LG 18650HG2 Li-ion Battery Data"
INSTITUTION = "McMaster University"
TEMPERATURE_FOLDER = "25degC"
ROOT_FILES = (
    "Readme file - Description of Experimental Tests.txt",
    "Technical Information and Experimental Test Results for LG 18650HG2.pdf",
)
FOLDER_FILES = (
    "549_HPPC.csv",
    "549_C20DisCh.csv",
    "549_Charge.csv",
    "549_Dis_0p5C.csv",
    "549_Dis_2C.csv",
    "551_Cap_1C.csv",
    "552_Cap_1C.csv",
    "551_UDDS.csv",
    "551_HWFET.csv",
    "551_LA92.csv",
    "551_US06.csv",
)
CSV_HEADER_START = "Time Stamp,"
CSV_COLUMNS = ("Time Stamp", "Step", "Status", "Prog Time", "Step Time", "Voltage", "Current")
TIMEOUT_S = 120
ATTEMPTS = 4
RETRY_DELAY_S = 5.0
USER_AGENT = "curl/8.0"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class RemoteFile:
    name: str
    folder: str
    size_bytes: int
    sha256: str
    url: str


class FetchError(Exception):
    pass


def get(url: str) -> tuple[int, bytes]:
    request = Request(url, headers={"Accept": "*/*", "User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with urlopen(request, timeout=TIMEOUT_S) as response:
                return response.status, response.read()
        except (URLError, ConnectionError, TimeoutError) as error:
            last_error = error
            sleep(RETRY_DELAY_S * (attempt + 1))
    raise FetchError(f"request failed for {url} after {ATTEMPTS} attempts: {last_error}")


def get_json(url: str) -> object:
    status, body = get(url)
    if status != 200:
        raise FetchError(f"HTTP {status} for {url}")
    return json.loads(body)


def list_files(folder_id: str, folder_name: str) -> list[RemoteFile]:
    query = urlencode({"folder_id": folder_id, "version": DATASET_VERSION})
    entries = get_json(f"{API_ROOT}/{DATASET_ID}/files?{query}&$start=0&$limit=1000")
    if not isinstance(entries, list):
        raise FetchError(f"unexpected listing for folder {folder_name}")
    return [
        RemoteFile(
            name=entry["filename"],
            folder=folder_name,
            size_bytes=int(entry["content_details"]["size"]),
            sha256=entry["content_details"]["sha256_hash"],
            url=entry["content_details"]["download_url"],
        )
        for entry in entries
    ]


def temperature_folder_id() -> str:
    folders = get_json(f"{API_ROOT}/{DATASET_ID}/folders/{DATASET_VERSION}")
    if not isinstance(folders, list):
        raise FetchError("unexpected folder listing")
    matches = [folder["id"] for folder in folders if folder["name"] == TEMPERATURE_FOLDER]
    if len(matches) != 1:
        names = [folder["name"] for folder in folders]
        raise FetchError(f"folder {TEMPERATURE_FOLDER} not found among {names}")
    return matches[0]


def select(listing: list[RemoteFile], wanted: tuple[str, ...]) -> list[RemoteFile]:
    by_name = {item.name: item for item in listing}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise FetchError(f"files missing from the remote listing: {missing}")
    return [by_name[name] for name in wanted]


def download(item: RemoteFile, destination: Path) -> dict[str, object]:
    status, body = get(item.url)
    digest = hashlib.sha256(body).hexdigest()
    if status != 200:
        raise FetchError(f"HTTP {status} for {item.name}")
    if len(body) != item.size_bytes:
        raise FetchError(f"{item.name}: got {len(body)} bytes, listing says {item.size_bytes}")
    if digest != item.sha256:
        raise FetchError(f"{item.name}: sha256 {digest} differs from listing {item.sha256}")
    destination.write_bytes(body)
    return {
        "name": item.name,
        "folder": item.folder,
        "http_status": status,
        "size_bytes": len(body),
        "sha256": digest,
        "listing_sha256": item.sha256,
        "download_url": item.url,
        "parse_check": parse_check(item.name, body),
    }


def decode(body: bytes, name: str) -> tuple[str, str]:
    for encoding in ("utf-8", "cp1252"):
        try:
            return body.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise FetchError(f"{name}: cannot decode as utf-8 or cp1252")


def parse_check(name: str, body: bytes) -> str:
    if name.endswith(".pdf"):
        if not body.startswith(b"%PDF"):
            raise FetchError(f"{name}: not a PDF")
        return "pdf header present"
    text, encoding = decode(body, name)
    if name.endswith(".txt"):
        if not text.strip():
            raise FetchError(f"{name}: empty text")
        return f"{encoding} text, {len(text.splitlines())} lines"
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(CSV_HEADER_START)]
    if len(starts) != 1:
        raise FetchError(f"{name}: expected one column header line, found {len(starts)}")
    header = [cell.strip() for cell in lines[starts[0]].split(",")]
    missing = [column for column in CSV_COLUMNS if column not in header]
    if missing:
        raise FetchError(f"{name}: columns {missing} missing from header {header}")
    rows = [line for line in lines[starts[0] + 2 :] if line.strip()]
    last = rows[-1].split(",")
    float(last[header.index("Voltage")])
    float(last[header.index("Current")])
    return f"csv, {len(rows)} data rows, {len(header) - 1} columns"


def main() -> int:
    try:
        root_id = "root"
        folder_id = temperature_folder_id()
        wanted = select(list_files(root_id, "root"), ROOT_FILES) + select(
            list_files(folder_id, TEMPERATURE_FOLDER), FOLDER_FILES
        )
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        records = [download(item, RAW_DIR / item.name) for item in wanted]
    except FetchError as error:
        print(f"FETCH FAILED: {error}")
        return 1
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": {
            "title": TITLE,
            "doi": DOI,
            "version": DATASET_VERSION,
            "licence": LICENCE,
            "institution": INSTITUTION,
            "landing_page": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
            "api_root": API_ROOT,
        },
        "files": records,
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", newline="\n")
    total = sum(int(record["size_bytes"]) for record in records)
    print(f"downloaded {len(records)} files, {total} bytes, all sizes and sha256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
