"""Download sanitation-related 311 requests from NYC Open Data and save a cleaned file."""

import time
import requests
import pandas as pd

from config import BASE_311_URL, SANITATION_COMPLAINTS, START_DATE, PAGE_LIMIT, RAW_311_FILE


SELECT_FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "status",
    "borough",
    "incident_zip",
    "latitude",
    "longitude",
    "community_board",
]


def build_where_clause() -> str:
    complaint_list = ",".join([f"'{c}'" for c in SANITATION_COMPLAINTS])
    return (
        f"created_date >= '{START_DATE}' "
        f"AND complaint_type in ({complaint_list})"
    )


def fetch_all_rows() -> pd.DataFrame:
    rows = []
    offset = 0

    while True:
        params = {
            "$select": ",".join(SELECT_FIELDS),
            "$where": build_where_clause(),
            "$limit": PAGE_LIMIT,
            "$offset": offset,
            "$order": "created_date ASC",
        }
        response = requests.get(BASE_311_URL, params=params, timeout=60)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        rows.extend(batch)
        print(f"Fetched {len(batch)} rows (total so far: {len(rows)})")

        if len(batch) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT
        time.sleep(0.2)

    return pd.DataFrame(rows)


def clean_311(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["community_board"] = df["community_board"].astype("string").str.strip()
    df["borough"] = df["borough"].astype("string").str.strip()
    df["incident_zip"] = df["incident_zip"].astype("string").str.strip()

    df["month"] = df["created_date"].dt.to_period("M").astype(str)
    df["year"] = df["created_date"].dt.year
    df["response_hours"] = (
        (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    )

    return df.sort_values("created_date").reset_index(drop=True)


def main() -> None:
    df = fetch_all_rows()
    print(f"Downloaded shape: {df.shape}")

    cleaned = clean_311(df)
    print(f"Cleaned shape: {cleaned.shape}")

    cleaned.to_parquet(RAW_311_FILE, index=False)
    print(f"Saved cleaned 311 data to {RAW_311_FILE}")
    print("Complaint counts:")
    print(cleaned["complaint_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
