"""
Fetch displacement data from the IDMC API.

Extracted from IDMC.ipynb for automated pipeline use.
Can also be run standalone: python scripts/fetch_data.py
"""

import requests
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    IDMC_API_URL, IDMC_CLIENT_ID, EAST_AFRICA_ISO3, INPUT_DIR
)


def fetch_idmc_data(
    api_url: str = IDMC_API_URL,
    client_id: str = IDMC_CLIENT_ID,
    iso3_codes: list = None,
) -> pd.DataFrame:
    """
    Fetch IDMC displacement data from the API.

    Returns filtered DataFrame with displacement records for East Africa.
    """
    if iso3_codes is None:
        iso3_codes = EAST_AFRICA_ISO3

    url = f"{api_url}?client_id={client_id}"
    print("Fetching data from IDMC API...")
    r = requests.get(url)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data)
    print(f"Total records from API: {len(df)}")

    filtered_df = df[df['iso3'].isin(iso3_codes)].copy()
    print(f"Records after filtering for East Africa: {len(filtered_df)}")

    min_year = filtered_df['year'].min()
    max_year = filtered_df['year'].max()
    print(f"Year range: {min_year} to {max_year}")

    return filtered_df


def save_fetched_data(df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Save fetched data to Excel.

    Defaults to Input/idmc_idus.xlsx (the canonical input location).
    """
    if output_path is None:
        output_path = INPUT_DIR / 'idmc_idus.xlsx'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(output_path), index=False)
    print(f"Data saved to: {output_path}")
    return output_path


def main():
    """Fetch IDMC data and save to Input/ directory."""
    df = fetch_idmc_data()
    save_fetched_data(df)
    print(f"\nFetch complete. {len(df)} records saved.")


if __name__ == '__main__':
    main()
