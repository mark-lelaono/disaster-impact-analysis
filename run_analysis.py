#!/usr/bin/env python3
"""
Disaster Displacement Analysis Pipeline

Unified CLI entry point that runs the full analysis pipeline or individual steps.

Usage:
    # Full pipeline (includes HTML report generation)
    python run_analysis.py --season OND --year 2025

    # Skip API fetch (use existing local data)
    python run_analysis.py --season OND --year 2025 --skip-fetch

    # Individual steps
    python run_analysis.py --season OND --year 2025 --step fetch
    python run_analysis.py --season OND --year 2025 --step process
    python run_analysis.py --season OND --year 2025 --step analyze
    python run_analysis.py --season OND --year 2025 --step seasonal
    python run_analysis.py --season OND --year 2025 --step store
    python run_analysis.py --season OND --year 2025 --step report

    # List previous runs
    python run_analysis.py --list-runs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_season_config, INPUT_DIR, OUTPUT_DIR,
    SHAPEFILE_PATH, ICONS_DIR, DB_PATH, DATA_DIR
)
from db import init_db, save_analysis_run, list_runs


VALID_STEPS = ['fetch', 'process', 'analyze', 'seasonal', 'store', 'report']

REPORTS_DIR = PROJECT_ROOT / 'reports'


def get_year_output_dir(year: int) -> Path:
    """Return the year-specific output directory: output/{year}/."""
    return OUTPUT_DIR / str(year)


def step_fetch():
    """Step 1: Fetch data from IDMC API."""
    print("\n" + "=" * 60)
    print("STEP 1: Fetching data from IDMC API")
    print("=" * 60)
    from scripts.fetch_data import fetch_idmc_data, save_fetched_data

    df = fetch_idmc_data()
    save_fetched_data(df)
    print(f"Fetch complete. {len(df)} records saved.")


def step_process(year: int, season: str, year_output_dir: Path) -> Path:
    """Step 2: Process raw data into summary Excel."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Processing data for {season} {year}")
    print("=" * 60)
    from scripts.process_idmc_data import process_idmc_data

    season_cfg = get_season_config(season)

    input_file = INPUT_DIR / 'idmc_idus.xlsx'
    year_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = year_output_dir / f'{season}_SummaryDATA_idmc_idus.xlsx'

    # ANNUAL: no month filter (include all months for the year)
    if season.upper() == 'ANNUAL':
        months = None
    else:
        months = season_cfg['months']

    process_idmc_data(
        str(input_file), str(output_file),
        year=year, months=months, season_name=season
    )
    print(f"Processing complete. Output: {output_file}")
    return output_file


def step_analyze(year: int, season: str, year_output_dir: Path):
    """Step 3: Run general (full-year) visualizations."""
    print("\n" + "=" * 60)
    print(f"STEP 3: Running general analysis for {year}")
    print("=" * 60)
    from scripts.displacement_analysis import (
        get_latest_summary_file, load_data,
        plot_geographic_distribution, plot_displacement_type_comparison,
        plot_top_events, plot_displacement_trend, plot_displacement_trend_2025,
    )
    from scripts.disaster_map_with_donuts_drm import plot_disaster_map_with_donuts

    season_cfg = get_season_config(season) if season else None

    summary_file = get_latest_summary_file(year_output_dir)
    df, dff = load_data(summary_file, INPUT_DIR)

    plot_geographic_distribution(df, SHAPEFILE_PATH, year_output_dir)
    plot_displacement_type_comparison(df, year_output_dir)
    plot_top_events(df, year_output_dir)
    plot_displacement_trend(dff, year_output_dir, year=year, season_cfg=season_cfg)
    plot_displacement_trend_2025(dff, year_output_dir, year=year, season_cfg=season_cfg)
    plot_disaster_map_with_donuts(
        shapefile_path=SHAPEFILE_PATH,
        icons_dir=ICONS_DIR,
        output_dir=year_output_dir,
        season=season,
        year=year,
        pipeline_df=df,
    )
    print("General analysis complete.")


def step_seasonal(year: int, season: str, year_output_dir: Path):
    """Step 4: Run season-specific visualizations. Returns the filtered DataFrame."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Running {season} {year} seasonal analysis")
    print("=" * 60)
    from scripts.displacement_analysis_season import run_seasonal_analysis

    df_season = run_seasonal_analysis(season, year, output_dir=year_output_dir)
    print(f"Seasonal analysis complete. {len(df_season)} records processed.")
    return df_season


def step_store(season: str, year: int, df, steps_executed: list) -> int:
    """Step 5: Store analysis results in SQLite database. Returns run_id."""
    print("\n" + "=" * 60)
    print("STEP 5: Storing results in database")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)

    run_id = save_analysis_run(
        db_path=DB_PATH,
        season=season,
        year=year,
        df=df,
        steps_executed=steps_executed,
    )
    print(f"Results stored. Run ID: {run_id}")
    print(f"Database: {DB_PATH}")
    return run_id


def step_report(season: str, year: int, year_output_dir: Path, pdf: bool = False) -> Path:
    """Step 6: Generate HTML report (and optionally PDF) from analysis outputs."""
    print("\n" + "=" * 60)
    print(f"STEP 6: Generating {season} {year} HTML report" + (" + PDF" if pdf else ""))
    print("=" * 60)
    from scripts.generate_report import generate_report

    report_path = generate_report(
        season=season,
        year=year,
        output_dir=year_output_dir,
        report_dir=REPORTS_DIR,
        pdf=pdf,
    )
    print(f"Report generated: {report_path}")
    return report_path


def run_full_pipeline(season: str, year: int, skip_fetch: bool = False, pdf: bool = False):
    """Run the complete pipeline: fetch -> process -> analyze -> seasonal -> store -> report."""
    year_output_dir = get_year_output_dir(year)

    print("=" * 60)
    print("DISASTER DISPLACEMENT ANALYSIS PIPELINE")
    print(f"Season: {season} | Year: {year}")
    print(f"Output: {year_output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    steps_executed = []

    # Step 1: Fetch
    if not skip_fetch:
        step_fetch()
        steps_executed.append('fetch')
    else:
        print("\nSkipping data fetch (using existing local data).")

    # Step 2: Process
    step_process(year, season, year_output_dir)
    steps_executed.append('process')

    # Step 3: General analysis
    step_analyze(year, season, year_output_dir)
    steps_executed.append('analyze')

    # Step 4: Seasonal analysis
    df_season = step_seasonal(year, season, year_output_dir)
    steps_executed.append('seasonal')

    # Step 5: Store in DB
    step_store(season, year, df_season, steps_executed)
    steps_executed.append('store')

    # Step 6: Generate HTML report (+ optional PDF)
    report_path = step_report(season, year, year_output_dir, pdf=pdf)
    steps_executed.append('report')

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps executed: {', '.join(steps_executed)}")
    print(f"All outputs in: {year_output_dir}")
    print(f"Report: {report_path}")
    print(f"Database: {DB_PATH}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Disaster Displacement Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --season OND --year 2025
  python run_analysis.py --season MAM --year 2025 --skip-fetch
  python run_analysis.py --season JJAS --year 2025 --step seasonal
  python run_analysis.py --season ANNUAL --year 2025 --step report
  python run_analysis.py --season ANNUAL --year 2025 --step report --pdf
  python run_analysis.py --season OND --year 2025 --pdf
  python run_analysis.py --list-runs
        """
    )

    parser.add_argument(
        '--season', choices=['OND', 'MAM', 'JJAS', 'ANNUAL'],
        help='Season to analyze (use ANNUAL for full-year analysis)'
    )
    parser.add_argument(
        '--year', type=int,
        help='Year to analyze'
    )
    parser.add_argument(
        '--step', choices=VALID_STEPS,
        help='Run a single step instead of the full pipeline'
    )
    parser.add_argument(
        '--skip-fetch', action='store_true',
        help='Skip API fetch and use existing local data'
    )
    parser.add_argument(
        '--list-runs', action='store_true',
        help='List all previous analysis runs from the database'
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='Also generate a PDF version of the report'
    )

    args = parser.parse_args()

    # Handle --list-runs
    if args.list_runs:
        init_db(DB_PATH)
        runs = list_runs(DB_PATH)
        if runs.empty:
            print("No analysis runs recorded yet.")
        else:
            print(runs.to_string(index=False))
        return

    # Validate required args
    if not args.season or not args.year:
        parser.error("--season and --year are required (unless using --list-runs)")

    year_output_dir = get_year_output_dir(args.year)

    # Run single step or full pipeline
    if args.step:
        if args.step == 'fetch':
            step_fetch()
        elif args.step == 'process':
            step_process(args.year, args.season, year_output_dir)
        elif args.step == 'analyze':
            step_analyze(args.year, args.season, year_output_dir)
        elif args.step == 'seasonal':
            df = step_seasonal(args.year, args.season, year_output_dir)
            # Auto-store after seasonal analysis
            init_db(DB_PATH)
            step_store(args.season, args.year, df, [args.step])
        elif args.step == 'store':
            # Store requires running seasonal first to get the DataFrame
            print("Note: 'store' step requires seasonal analysis data. Running seasonal + store.")
            df = step_seasonal(args.year, args.season, year_output_dir)
            step_store(args.season, args.year, df, ['seasonal', 'store'])
        elif args.step == 'report':
            try:
                step_report(args.season, args.year, year_output_dir, pdf=args.pdf)
            except FileNotFoundError as e:
                print(f"\nError: {e}")
                sys.exit(1)
    else:
        run_full_pipeline(args.season, args.year, skip_fetch=args.skip_fetch, pdf=args.pdf)


if __name__ == '__main__':
    main()
