from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "neighborhood_month.parquet"

df = pd.read_parquet(DATA_FILE)
print(DATA_FILE)
print(df.columns.tolist())
print(df.head(10).to_string())
