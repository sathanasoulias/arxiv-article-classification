"""Shared configuration: paths, taxonomy mapping, and experiment constants.

Every script imports from here, so there are no hard-coded paths elsewhere.

Layout rationale:
  - The project folder is inside OneDrive, so large artifacts must not live there.
    Raw data and derived data (samples, embeddings) go to WORK_DIR under LOCALAPPDATA.
  - Small results (stats, metrics, figures, sample ids) stay in the repo and are committed.
"""

import os
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Derived data: large and regenerable, kept out of the synced project folder.
WORK_DIR = Path(os.environ["LOCALAPPDATA"]) / "arxiv-iris"

# The Kaggle dataset is re-published monthly; pinning the version is what makes the
# committed sample ids meaningful.
DATASET_VERSION = "309"
DATA_FILE = (
    Path(os.environ["LOCALAPPDATA"])
    / "kagglehub/datasets/Cornell-University/arxiv/versions"
    / DATASET_VERSION
    / "arxiv-metadata-oai-snapshot.json"
)

# --- Preprocessing -----------------------------------------------------------------

YEAR_MIN = 2016           # removes all 24 legacy archives; rebalances physics vs cs
MIN_ABSTRACT_WORDS = 20   # drops 15,567 stub abstracts
MAX_TOKENS = 256          # fits 67.2% of abstracts whole; 128 would fit only 20.5%

# --- Sample sizes ------------------------------------------------------------------
# Two phases. The pilot matches Arcan (2026) for direct comparison and runs on CPU;
# the main phase is sized so the rarest archive still has enough papers.

PHASES = {
    "pilot": {"total": 10_000, "cluster": 5_000, "floor": 100},
    "main": {"total": 150_000, "cluster": 50_000, "floor": 0},
}

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / val / test

# --- Embedding models --------------------------------------------------------------
# Chosen from Arcan (2026), who benchmarked all four candidates on this exact task:
#   all-mpnet-base-v2   ARI 0.470  <- best alignment with arXiv categories
#   all-MiniLM-L6-v2    ARI 0.442     ~5x faster, close second
#   allenai-specter     ARI 0.415     best silhouette (0.095) but worse alignment
#   scibert             ARI 0.304     worst; never trained for sentence similarity
# The two science-specific models lost, so we use the general-purpose pair. MiniLM
# runs first to validate the pipeline quickly; MPNet produces the reported numbers.

EMBEDDING_MODELS = {
    "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384},
    "mpnet": {"name": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
}
EMBED_BATCH_SIZE = 64

# --- arXiv taxonomy ----------------------------------------------------------------
# arXiv puts all of physics under one group but splits it into many archives.
# Neither level is a correct reference on its own, so results are reported against both:
#   - archives (20 from 2016+) handle physics correctly but lump all of cs together
#   - groups (8) are coarser still, treating 13 physics fields as one class

PHYSICS_ARCHIVES = {
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph",
}

GROUPS = {
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
}


def archive_of(category: str) -> str:
    """'astro-ph.GA' -> 'astro-ph';  'cs.LG' -> 'cs'."""
    return category.split(".", 1)[0]


def group_of(category: str) -> str:
    """Collapse a category to one of the 8 top-level arXiv groups."""
    archive = archive_of(category)
    if archive in PHYSICS_ARCHIVES:
        return "physics"
    if archive in GROUPS:
        return archive
    return "other"


def work_path(*parts) -> Path:
    """Path inside WORK_DIR, creating parent directories as needed."""
    p = WORK_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
