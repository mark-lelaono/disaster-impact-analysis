# Disaster Impact Analysis

A data pipeline for analyzing and visualizing disaster-related displacement data. The project fetches data from external APIs, processes it into summary datasets, generates visualizations, and stores results in a SQLite database.

## Directory Structure

```
├── run_analysis.py         # CLI pipeline orchestrator
├── config.py               # Central configuration (paths, seasons, DB)
├── db.py                   # SQLite database layer
├── scripts/
│   ├── fetch_data.py                       # Step 1: API data fetcher
│   ├── process_idmc_data.py                # Step 2: Data processing
│   ├── displacement_analysis.py            # Step 3: General visualizations
│   ├── displacement_analysis_season.py     # Step 4: Season-aware analysis
│   ├── disaster_map_with_donuts_drm.py     # Standalone: DRM impact map
│   └── *.ipynb                             # Interactive notebooks
├── Input/                  # Raw input data files
├── output/                 # Generated visualizations and summary data
├── shapefiles/             # Geospatial boundary files
├── icons/                  # Disaster type icons (PNG/SVG)
└── data/                   # SQLite database (auto-created)
```

## Data Pipeline

```
External API
    │
    ▼
[Step 1] Fetch raw displacement data
    │
    ▼
[Step 2] Process and summarize data (season-aware)
    │
    ├─────────────────────────────┐
    ▼                             ▼
[Step 3] General visualizations  [Step 4] Seasonal analysis
                                  │
                                  ▼
                       [Step 5] Store results in SQLite DB
```

## Quick Start

### Full pipeline (single command)

```bash
python run_analysis.py --season OND --year 2025
```

### Skip API fetch (use existing local data)

```bash
python run_analysis.py --season OND --year 2025 --skip-fetch
```

### Run individual steps

```bash
python run_analysis.py --season OND --year 2025 --step fetch
python run_analysis.py --season OND --year 2025 --step process
python run_analysis.py --season OND --year 2025 --step analyze
python run_analysis.py --season OND --year 2025 --step seasonal
```

### Query previous runs

```bash
python run_analysis.py --list-runs
```

## Supported Seasons

| Code | Months                        |
|------|-------------------------------|
| OND  | October, November, December   |
| JJAS | June, July, August, September |
| MAM  | March, April, May             |

## Database

Analysis results are automatically stored in `data/analysis.db` (SQLite) after each run. Three tables:

- **analysis_runs** - Metadata per pipeline execution (season, year, record counts)
- **displacement_records** - Full processed records for each run
- **run_summary** - Aggregated statistics by country, month, and type

## Dependencies

```bash
pip install pandas matplotlib seaborn plotly geopandas numpy requests cairosvg pillow openpyxl
```
