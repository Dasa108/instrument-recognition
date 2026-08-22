"""Download and extract the IRMAS dataset into data/raw/.

Spec: spec.md Section 3 (Dataset) — IRMAS, ~9k x 3s clips, 11 instruments, via Zenodo
(DOI 10.5281/zenodo.1290750).
Milestone: spec.md Section 9, #1.

Downloads both Training data (single-label, 3s clips — what Phase 1 trains on) and Testing data
(multi-labeled, variable-length — NOT a drop-in single-label test set; Phase 1's val/test split is
carved from Training data instead, grouped by song. See DECISIONS.md, "IRMAS download scope"
entry). Testing data is fetched now so it's available later for Phase 2 / bonus eval without a
second multi-GB download.

Each archive is verified against Zenodo's published MD5 checksum before extraction; re-running this
script skips any file that's already downloaded and verified.
"""

import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ARCHIVE_DIR = DATA_DIR / "_archives"

# filename -> (download URL, expected MD5, extract-to subdirectory)
IRMAS_FILES = {
    "IRMAS-TrainingData.zip": (
        "https://zenodo.org/api/records/1290750/files/IRMAS-TrainingData.zip/content",
        "4fd9f5ed5a18d8e2687e6360b5f60afe",
        DATA_DIR / "IRMAS-TrainingData",
    ),
    "IRMAS-TestingData-Part1.zip": (
        "https://zenodo.org/api/records/1290750/files/IRMAS-TestingData-Part1.zip/content",
        "5a2e65520dcedada565dff2050bb2a56",
        DATA_DIR / "IRMAS-TestingData",
    ),
    "IRMAS-TestingData-Part2.zip": (
        "https://zenodo.org/api/records/1290750/files/IRMAS-TestingData-Part2.zip/content",
        "afb0c8ea92f34ee653693106be95c895",
        DATA_DIR / "IRMAS-TestingData",
    ),
    "IRMAS-TestingData-Part3.zip": (
        "https://zenodo.org/api/records/1290750/files/IRMAS-TestingData-Part3.zip/content",
        "9b3fb2d0c89cdc98037121c25bd5b556",
        DATA_DIR / "IRMAS-TestingData",
    ),
}


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def ensure_downloaded(filename: str, url: str, expected_md5: str) -> Path:
    archive_path = ARCHIVE_DIR / filename
    if archive_path.exists() and md5sum(archive_path) == expected_md5:
        print(f"[skip] {filename} already downloaded and verified")
        return archive_path

    print(f"[download] {filename}")
    download_file(url, archive_path)

    actual_md5 = md5sum(archive_path)
    if actual_md5 != expected_md5:
        archive_path.unlink()
        raise ValueError(
            f"{filename}: MD5 mismatch (expected {expected_md5}, got {actual_md5}) — "
            "deleted, re-run to retry."
        )
    print(f"[ok] {filename} checksum verified")
    return archive_path


def extract(archive_path: Path, extract_to: Path) -> None:
    # Marker is per-archive (not per destination dir): IRMAS-TestingData-Part1/2/3 all extract
    # into the *same* data/raw/IRMAS-TestingData/ directory, so a shared marker there would make
    # the 2nd and 3rd extraction silently no-op after the 1st (this actually happened once — see
    # DECISIONS.md, "IRMAS extraction bug" entry).
    marker = archive_path.with_suffix(archive_path.suffix + ".extracted")
    if marker.exists():
        print(f"[skip] {archive_path.name} already extracted")
        return
    print(f"[extract] {archive_path.name} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_to)
    marker.touch()


def main() -> None:
    for filename, (url, expected_md5, extract_to) in IRMAS_FILES.items():
        archive_path = ensure_downloaded(filename, url, expected_md5)
        extract(archive_path, extract_to)
    print("Done. Training data: data/raw/IRMAS-TrainingData/ ; "
          "Testing data: data/raw/IRMAS-TestingData/")


if __name__ == "__main__":
    main()
