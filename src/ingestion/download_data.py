"""RetailPulse — download the raw datasets from the UCI ML Repository.

Reproducible replacement for the Kaggle links in the project brief: the Kaggle
"Online Retail II" dataset is a mirror of UCI id 502, and UCI allows direct
download without an API key, so the pipeline can bootstrap itself.

Usage:
    python -m src.ingestion.download_data
    python -m src.ingestion.download_data --include-retail-i
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg


SOURCES = {
    "online_retail_ii": {
        "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "dest": cfg.RAW / "online_retail_ii",
        "expected": "online_retail_II.xlsx",
    },
    "online_retail_i": {
        "url": "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        "dest": cfg.RAW / "online_retail_i",
        "expected": "Online Retail.xlsx",
    },
}


def download(name: str, include_retail_i: bool = False) -> Path:
    spec = SOURCES[name]
    dest: Path = spec["dest"]
    target = dest / spec["expected"]

    if target.exists():
        print(f"[download] {name}: already present -> {target}")
        return target

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = cfg.RAW / f"{name}.zip"
    print(f"[download] {name}: fetching {spec['url']}")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as fh:
        fh.write(resp.read())
    print(f"[download] {name}: {zip_path.stat().st_size / 1_048_576:.1f} MB")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f"[download] {name}: extracted -> {target}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description="Download UCI Online Retail datasets")
    ap.add_argument("--include-retail-i", action="store_true",
                    help="also fetch the smaller 2010-2011 dataset (UCI id 352)")
    args = ap.parse_args()

    download("online_retail_ii")
    if args.include_retail_i:
        download("online_retail_i", include_retail_i=True)
    print("\nRaw data ready. Next: python -m src.ingestion.load_data")


if __name__ == "__main__":
    main()
