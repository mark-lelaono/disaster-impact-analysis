# Disaster Impact Analysis

A data pipeline for analyzing and visualizing disaster-related displacement in Eastern Africa. Fetches data from the IDMC API, processes it by season, generates visualizations (maps, charts, heatmaps), and stores results in SQLite.

## Directory Structure

```
├── run_analysis.py                             # CLI pipeline orchestrator
├── config.py                                   # Central configuration (paths, seasons, countries)
├── db.py                                       # SQLite database layer
├── scripts/
│   ├── fetch_data.py                           # Step 1: IDMC API data fetcher
│   ├── process_idmc_data.py                    # Step 2: Filter & summarize by season/year
│   ├── displacement_analysis.py                # Step 3: General visualizations
│   ├── displacement_analysis_season.py         # Step 4: Season-specific analysis
│   ├── disaster_map_with_donuts_drm.py         # Step 3: Disaster map with icons & donuts
│   └── generate_report.py                      # Step 6: HTML report generator
├── Input/                                      # Raw IDMC data (idmc_idus.xlsx)
├── output/                                     # Generated visualizations & summary Excel
│   └── {year}/                                 # Year-specific outputs (images, stats, Excel)
├── reports/                                    # Generated HTML & PDF reports (auto-created)
├── shapefiles/                                 # ICPAC country boundary shapefiles
├── icons/                                      # Humanitarian disaster icons (256x256 PNG)
└── data/                                       # SQLite database (auto-created)
```

## Data Pipeline

```
IDMC API
    │
    ▼
[Step 1] Fetch raw displacement data (8,000+ records, East Africa)
    │
    ▼
[Step 2] Process & filter by season + year → summary Excel
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[Step 3] General visualizations     [Step 4] Seasonal analysis
  • Country bar charts                • Season-prefixed charts
  • Choropleth maps                   • Disaster type breakdown
  • Displacement trends               • Pre-season + season trends
  • Disaster map with icons           • Country-month heatmap
    & donut charts                    • Summary statistics (.txt)
    │                                  │
    └──────────┬───────────────────────┘
               ▼
        [Step 5] Store results in SQLite
               │
               ▼
        [Step 6] Generate HTML report (+ optional PDF)
          • Professional cover page
          • Executive summary with key figures
          • Embedded maps, charts & tables
          • Seasonal comparison (ANNUAL mode)
          • Data sources & methodology
          → reports/{season}_displacement_report_{year}.html
          → reports/{season}_displacement_report_{year}.pdf  (with --pdf)
```

## Quick Start

### Full pipeline (season-specific)

```bash
python run_analysis.py --season OND --year 2025
python run_analysis.py --season OND --year 2025 --pdf     # also generate PDF
```

### Full-year analysis

```bash
python run_analysis.py --season ANNUAL --year 2024
python run_analysis.py --season ANNUAL --year 2024 --pdf  # also generate PDF
```

### Skip API fetch (use cached data)

```bash
python run_analysis.py --season MAM --year 2020 --skip-fetch
python run_analysis.py --season ANNUAL --year 2024 --skip-fetch
```

### Run a single step

```bash
python run_analysis.py --season OND --year 2025 --step fetch
python run_analysis.py --season OND --year 2025 --step process
python run_analysis.py --season OND --year 2025 --step analyze
python run_analysis.py --season OND --year 2025 --step seasonal
python run_analysis.py --season OND --year 2025 --step store
python run_analysis.py --season OND --year 2025 --step report
```

### Generate only the HTML report (from existing outputs)

```bash
python run_analysis.py --season ANNUAL --year 2025 --step report
python run_analysis.py --season OND --year 2025 --step report
```

### Query previous runs

```bash
python run_analysis.py --list-runs
```

## Supported Seasons

| Code   | Season Months                        | Pre-Season Months          |
|--------|--------------------------------------|----------------------------|
| MAM    | March, April, May                    | January, February          |
| JJAS   | June, July, August, September        | March, April, May          |
| OND    | October, November, December          | July, August, September    |
| ANNUAL | January – December (Full Year)       | — (none)                   |

Pre-season months are used in displacement trend charts to show the build-up before the season. ANNUAL mode analyses all 12 months with no pre-season split.

## Output Visualizations

Each pipeline run generates outputs in `output/{year}/`, keeping each year's data isolated. For example, `python run_analysis.py --season OND --year 2025` writes to `output/2025/`. This ensures reports always reference the correct year's images and statistics.

### Step 3 — General Analysis
| File | Description |
|------|-------------|
| `displacements_by_country.png` | Bar chart of displacements by country |
| `displacement_choropleth_map.png` | Choropleth map of displacement intensity |
| `displacement_by_type_pie.png` | Pie chart: Conflict vs Disaster vs Other |
| `monthly_displacement_by_type.png` | Monthly stacked bar by displacement type |
| `top_10_events.png` | Top 10 displacement-causing events |
| `displacement_trend_over_time.png` | Historical monthly trend with season highlight |
| `displacement_trend_{year}_only.png` | Pre-season + season trend for the target year |
| `{season}_disaster_map_with_donuts.png` | Map with disaster icons + 3 donut charts |

### Step 4 — Seasonal / Annual Analysis
All files prefixed with the season code (e.g., `mam_`, `ond_`, `jjas_`, `annual_`):

| File | Description |
|------|-------------|
| `{season}_displacements_by_country.png` | Season-filtered country bar chart |
| `{season}_displacement_choropleth_map.png` | Season-filtered choropleth |
| `{season}_displacement_by_type_pie.png` | Season displacement type pie |
| `{season}_monthly_displacement_by_type.png` | Season monthly stacked bar |
| `{season}_displacements_by_disaster_type.png` | Disaster type breakdown bar |
| `{season}_disaster_type_pie.png` | Disaster type pie chart |
| `{season}_top_10_events.png` | Season top 10 events |
| `{season}_trend_across_years.png` | Season totals across all years |
| `{season}_total_by_year.png` | Year-over-year bar chart |
| `{season}_pre_season_trend_{year}.png` | Pre-season + season monthly trend (or full monthly trend for ANNUAL) |
| `{season}_country_month_heatmap.png` | Country vs month heatmap |
| `{season}_summary_statistics.txt` | Text summary with totals and breakdowns |

### Disaster Map Features
- Humanitarian icons (OCHA-style, green) placed inside country polygons
- Icons dynamically positioned using `representative_point()` + radial scatter
- Adaptive icon sizing based on number of hazards per country
- Legend with icon previews
- Three donut charts: Total Events, Total Displaced, Total Affected
- Values formatted as K/M (e.g., 1.2M, 287K)

### Step 6 — HTML & PDF Report

Each pipeline run generates a self-contained HTML report in `reports/`. Add `--pdf` to also produce a PDF version with the same styling:

| File | Description |
|------|-------------|
| `{season}_displacement_report_{year}.html` | Full analytical report with embedded images |
| `{season}_displacement_report_{year}.pdf` | PDF version (A4, with page numbers) — requires `--pdf` flag |

The report includes:
- **Cover page** with headline statistics (total displaced, events, countries, conflict share)
- **Executive summary** with key figures and critical findings
- **Methodology** section with seasonal framework and data source documentation
- **Full analysis** with all 12 generated maps/charts embedded as figures, plus country and disaster type data tables
- **Seasonal comparison** (ANNUAL reports only) — side-by-side cards and comparison table for MAM, JJAS, OND with their disaster maps
- **Data sources annex** listing all contributing organizations

The report generator (`scripts/generate_report.py`) can also be run standalone:

```bash
python scripts/generate_report.py --season ANNUAL --year 2025
python scripts/generate_report.py --season OND --year 2025
python scripts/generate_report.py --season OND --year 2025 --pdf   # HTML + PDF
```

## Database

Results are stored in `data/analysis.db` (SQLite) after each run:

| Table | Contents |
|-------|----------|
| `analysis_runs` | Metadata per pipeline run (season, year, timestamp, record counts) |
| `displacement_records` | Full processed records for each run |
| `run_summary` | Aggregated stats by country, month, and displacement type |

## Countries Covered

Somalia, Ethiopia, Kenya, Uganda, Tanzania, Sudan, South Sudan, Rwanda, Burundi, Djibouti, Eritrea

## Dependencies

```bash
pip install pandas matplotlib seaborn geopandas numpy requests openpyxl shapely

# Optional: for PDF report generation (--pdf flag)
pip install weasyprint
```
