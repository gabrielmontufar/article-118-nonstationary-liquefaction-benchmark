from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_designsafe"
NISQUALLY = RAW / "nisqually_designsafe"
CANTERBURY = RAW / "canterbury_designsafe_PRJ-2937"

LISTING_BASE = "https://www.designsafe-ci.org/api/datafiles/tapis/public/listing/designsafe.storage.published"
DOWNLOAD_URL = "https://www.designsafe-ci.org/api/datafiles/tapis/public/download/designsafe.storage.published/?doi="
REFERER = "https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3758"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_path(session: requests.Session, path: str) -> list[dict]:
    url = f"{LISTING_BASE}/{quote(path, safe='/')}"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()["listing"]


def get_download_href(session: requests.Session, headers: dict[str, str], path: str) -> str:
    response = session.put(DOWNLOAD_URL, headers=headers, json={"paths": [path]}, timeout=60)
    response.raise_for_status()
    return response.json()["href"]


def download(session: requests.Session, headers: dict[str, str], path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    href = get_download_href(session, headers, path)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(href, stream=True, timeout=180) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def main() -> None:
    session = requests.Session()
    response = session.get(REFERER, timeout=60)
    response.raise_for_status()
    csrf = session.cookies.get("csrftoken")
    if not csrf:
        raise RuntimeError("DesignSafe CSRF token was not provided by the public data page")
    headers = {"Referer": REFERER, "X-CSRFToken": csrf, "Content-Type": "application/json"}

    download(session, headers, "/PRJ-3758/Summary Table S1.xlsx", NISQUALLY / "Summary Table S1.xlsx")
    for item in list_path(session, "PRJ-3758/Case History Data"):
        if item["type"] == "file" and item["name"].lower().endswith(".xlsx"):
            download(session, headers, item["path"], NISQUALLY / item["name"])
    download(session, headers, "/PRJ-2937/CANTERBURYDATASET.mat", CANTERBURY / "CANTERBURYDATASET.mat")

    manifest = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file() and path.name != "download_manifest.json":
            manifest.append(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (RAW / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(manifest), "total_bytes": sum(row["size_bytes"] for row in manifest)}, indent=2))


if __name__ == "__main__":
    main()
