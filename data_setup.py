import os
import urllib.request
from pathlib import Path

# Base Zenodo record URL
ZENODO_BASE = "https://zenodo.org/records/15680920/files"

FILES = [
    "DST_MEASUREMENTS.parquet",
    "GENOMES.parquet",
    "WGS_SAMPLES.parquet",
    "MUTATIONS.parquet",
    "VARIANTS.parquet",
]

DATA_DIR = Path("data/cryptic-tables-v3.4.0")

def download_data():
    """Download CRyPTIC Parquet data files from Zenodo."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = DATA_DIR / fname
        if not dest.exists():
            url = f"{ZENODO_BASE}/{fname}"
            print(f"Downloading {fname}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"Finished: {fname}")
            except Exception as e:
                print(f"Failed to download {fname}: {e}")
        else:
            print(f"{fname} already exists, skipping.")

    print("\nData available in:", DATA_DIR)


if __name__ == "__main__":
    download_data()
