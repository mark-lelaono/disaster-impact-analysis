"""
Script to process idmc_idus.xlsx and generate summary data file.

This script:
1. Reads the raw IDMC displacement data from idmc_idus.xlsx
2. Filters data for year 2025 (January - December)
3. Filters and retains specific columns
4. Extracts the Month from the displacement_date column
5. Renames 'type' column to 'Disaster_Type'
6. Outputs the summary data to output/<CurrentMonth>_SummaryDATA_idmc_idus.xlsx
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def process_idmc_data(input_file: str, output_file: str, year: int = 2025) -> pd.DataFrame:
    """
    Process IDMC displacement data and create summary file.

    Args:
        input_file: Path to the input Excel file (idmc_idus.xlsx)
        output_file: Path to the output Excel file (SummaryDATA_idmc_idus_May2025.xlsx)
        year: Year to filter data for (default: 2025)

    Returns:
        DataFrame containing the processed summary data
    """
    # Read the raw data
    print(f"Reading data from: {input_file}")
    df = pd.read_excel(input_file)
    print(f"Total records loaded: {len(df)}")

    # Convert displacement_date to datetime for filtering
    df['displacement_date'] = pd.to_datetime(df['displacement_date'], errors='coerce')

    # Filter for the specified year
    df_year = df[df['displacement_date'].dt.year == year].copy()
    print(f"Records for year {year}: {len(df_year)}")

    # Columns to retain (from original data)
    columns_to_keep = [
        'id',
        'country',
        'displacement_type',
        'type',  # Will be renamed to Disaster_Type
        'figure',
        'event_name',
        'sources',
        'source_url',
        'locations_name',
        'displacement_date'  # Will be used to extract Month
    ]

    # Check which columns exist in the data
    available_columns = [col for col in columns_to_keep if col in df_year.columns]
    missing_columns = [col for col in columns_to_keep if col not in df_year.columns]

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")

    # Filter to keep only required columns
    df_filtered = df_year[available_columns].copy()

    # Extract Month name from displacement_date
    df_filtered['Month'] = df_filtered['displacement_date'].dt.strftime('%B')

    # Rename 'type' column to 'Disaster_Type'
    if 'type' in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={'type': 'Disaster_Type'})

    # Select final columns for output (excluding displacement_date, keeping Month)
    final_columns = [
        'id',
        'country',
        'displacement_type',
        'Disaster_Type',
        'figure',
        'event_name',
        'sources',
        'source_url',
        'locations_name',
        'Month'
    ]

    # Keep only columns that exist
    final_columns = [col for col in final_columns if col in df_filtered.columns]
    df_summary = df_filtered[final_columns].copy()

    # Remove rows with missing Month (invalid dates)
    df_summary = df_summary.dropna(subset=['Month'])

    print(f"Records after processing: {len(df_summary)}")
    print(f"Columns in output: {list(df_summary.columns)}")

    # Save to Excel
    print(f"Saving summary data to: {output_file}")
    df_summary.to_excel(output_file, index=False)
    print("Processing complete!")

    return df_summary


def main():
    # Define file paths
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / 'Input'
    output_dir = script_dir.parent / 'output'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get current month name as prefix
    current_month = datetime.now().strftime('%B')

    input_file = input_dir / 'idmc_idus.xlsx'
    output_file = output_dir / f'{current_month}_SummaryDATA_idmc_idus.xlsx'

    # Process the data
    df_summary = process_idmc_data(str(input_file), str(output_file))

    # Display sample of the output
    print("\nSample of processed data:")
    print(df_summary.head(10))

    print("\nData info:")
    print(df_summary.info())

    print("\nMonth distribution:")
    print(df_summary['Month'].value_counts())


if __name__ == '__main__':
    main()
