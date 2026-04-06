from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# NYC Open Data 311 endpoint (2020-present)
BASE_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# First-pass sanitation-related complaint types.
# TODO: Update w/which complaint categories are most useful.
SANITATION_COMPLAINTS = [
    "Missed Collection",
    "Dirty Conditions",
    "Illegal Dumping",
    "Litter Basket / Request",
    "Overflowing Litter Baskets",
]

START_DATE = "2024-01-01T00:00:00"
PAGE_LIMIT = 50000

RAW_311_FILE = RAW_DIR / "311_sanitation_filtered.parquet"
FEATURE_FILE = PROCESSED_DIR / "neighborhood_month.parquet"
