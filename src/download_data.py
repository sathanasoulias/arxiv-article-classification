"""Download the arXiv metadata snapshot from Kaggle.

Requires KAGGLE_API_TOKEN in the environment. The file is ~5.6 GB, so the cache is
deliberately kept outside the OneDrive-synced project folder.

The newest dataset version can 404 while Kaggle is still compressing it server-side,
so this requests the single data file directly and falls back to older versions.
"""

import os
import sys
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is not set.")

os.environ.setdefault(
    "KAGGLEHUB_CACHE",
    str(Path(os.environ["LOCALAPPDATA"]) / "kagglehub"),
)

import kagglehub  # noqa: E402  (import after cache env is set)

FILENAME = "arxiv-metadata-oai-snapshot.json"
HANDLES = [
    "Cornell-University/arxiv",
    "Cornell-University/arxiv/versions/308",
    "Cornell-University/arxiv/versions/307",
]

for handle in HANDLES:
    print(f"Trying {handle} ...", flush=True)
    try:
        path = Path(kagglehub.dataset_download(handle, path=FILENAME))
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}\n", flush=True)
        continue
    print(f"\nDownloaded: {path}")
    print(f"Size: {path.stat().st_size / 1e9:.2f} GB")
    break
else:
    sys.exit("All download attempts failed.")
