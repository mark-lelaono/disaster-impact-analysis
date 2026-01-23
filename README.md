# IDMC Codebase Overview

## Directory Structure

```
IDMC_Codebase/
├── scripts/
│   ├── IDMC.ipynb                    # Step 1: Data fetching from IDMC API
│   ├── process_idmc_data.py          # Step 2: Generate summary data
│   ├── displacement_analysis.py      # Step 3: Generate visualizations
│   ├── displacement_analysis_ond.py  # Step 4: OND season analysis
│   ├── disaster_map_with_donuts_drm.py  # Step 5: DRM map with donut charts
│   ├── change_color.py               # Utility: Recolor icons (SVG/PNG)
│   └── Displacement_Analysis_2025.ipynb  # Interactive analysis notebook
│
├── Input/
│   ├── idmc_idus.xlsx                # Raw data from API (5,676 records)
│   └── SummaryDATA_idmc_idus_May2025.xlsx  # Processed summary (184 records)
│
├── shapefiles/
│   ├── Administrative0_Boundaries_ICPAC_Countries.shp
│   ├── Administrative0_Boundaries_ICPAC_Countries.shx
│   ├── Administrative0_Boundaries_ICPAC_Countries.dbf
│   └── Administrative0_Boundaries_ICPAC_Countries.prj
│
├── icons/
│   ├── Flood_icon.png
│   ├── Earthquake_icon.png
│   ├── Epidemic_icon.png
│   ├── Drought_icon.png
│   └── ... (other disaster type icons)
│
├── output/
│   ├── displacement_trend_red.png
│   ├── displacement_donut_charts.png
│   ├── ond_disaster_map_with_donuts.png
│   ├── ond_country_month_heatmap.png
│   └── ... (other visualizations)
│
└── html/
    └── Visualization.html
```

---

## Data Pipeline

The scripts should be executed in the following order:

```
IDMC API
    │
    ▼
[Step 1] scripts/IDMC.ipynb
    │
    ▼
Input/idmc_idus.xlsx (5,676 records)
    │
    ▼
[Step 2] scripts/process_idmc_data.py
    │
    ▼
output/<Month>_SummaryDATA_idmc_idus.xlsx
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[Step 3] displacement_analysis.py    [Step 4] displacement_analysis_ond.py
    │                                  │
    ▼                                  ▼
output/ (general visualizations)     output/ (OND-specific visualizations)
                                       │
                                       ▼
                            [Step 5] disaster_map_with_donuts_drm.py
                                       │
                                       ▼
                            output/ond_disaster_map_with_donuts.png
```

---

## Step 1: IDMC.ipynb - Data Extraction

**Purpose:** Fetches displacement data from IDMC API and saves to Excel

### Workflow:
1. **API Call** to `https://helix-tools-api.idmcdb.org/external-api/idus/all/?client_id=ICPACAUG10`
2. **Filter** for East African countries: `RWA, UGA, TZA, BDI, SSD, SDN, ERI, ETH, DJI, SOM, KEN`
3. **Output:** `Input/idmc_idus.xlsx` (5,676 records, 35 columns, years 2018-2025)

### Data Schema (35 columns):
| Column | Description |
|--------|-------------|
| `id`, `country`, `iso3` | Identifiers |
| `latitude`, `longitude`, `centroid` | Geographic coordinates |
| `displacement_type` | Disaster or Conflict |
| `figure` | Number of displaced people |
| `year`, `displacement_date`, `displacement_start_date`, `displacement_end_date` | Temporal data |
| `event_id`, `event_name`, `event_start_date`, `event_end_date` | Event details |
| `category`, `subcategory`, `type`, `subtype` | Classification |
| `sources`, `source_url` | Data provenance |
| `locations_name`, `locations_coordinates`, `locations_accuracy` | Location details |

---

## Step 2: process_idmc_data.py - Data Processing

**Purpose:** Processes raw IDMC data and generates the summary Excel file

### Workflow:
1. **Read** `Input/idmc_idus.xlsx` (raw data)
2. **Filter by year** (default: 2025)
3. **Filter columns** to retain:
   - `id`, `country`, `displacement_type`, `type` (renamed to `Disaster_Type`)
   - `figure`, `event_name`, `sources`, `source_url`, `locations_name`
4. **Extract Month** from `displacement_date` column
5. **Output:** `output/<Month>_SummaryDATA_idmc_idus.xlsx`

### Output Schema (10 columns):
| Column | Description |
|--------|-------------|
| `id` | Unique identifier |
| `country` | Country name |
| `displacement_type` | Disaster or Conflict |
| `Disaster_Type` | Specific type (e.g., Flood, armed conflict) |
| `figure` | Number of displaced people |
| `event_name` | Name of the displacement event |
| `sources` | Data source |
| `source_url` | URL to the source |
| `locations_name` | Location details |
| `Month` | Month extracted from displacement_date |

### Usage:
```bash
cd scripts
python process_idmc_data.py
```

---

## Step 3: displacement_analysis.py - General Visualizations

**Purpose:** Generates all displacement visualizations from the summary data

### Workflow:
1. **Read** the latest summary file from `output/` folder
2. **Read** raw data from `Input/idmc_idus.xlsx` for trend analysis
3. **Generate visualizations** and save to `output/` folder

### Generated Visualizations:
| Output File | Description |
|-------------|-------------|
| `displacements_by_country.png` | Bar chart of displacements by country (log scale) |
| `displacement_choropleth_map.png` | Choropleth map of displacement figures |
| `displacement_by_type_pie.png` | Pie chart of Disaster vs Conflict |
| `monthly_displacement_by_type.png` | Stacked bar chart by month and type |
| `top_10_events.png` | Top 10 events causing displacement |
| `displacement_trend_over_time.png` | Time series trend (2018-2025) |
| `displacement_trend_2025_only.png` | 2025 trend: Jan-Sep (blue), OND (red) |
| `eastern_africa_disaster_map_with_donuts.png` | Map with donut charts |

### Usage:
```bash
cd scripts
python displacement_analysis.py
```

---

## Step 4: displacement_analysis_ond.py - OND Season Analysis

**Purpose:** Generates displacement visualizations specifically for the OND (October-November-December) season

### Workflow:
1. **Read** the latest summary file from `output/` folder
2. **Filter** data for OND months only (October, November, December)
3. **Read** raw data from `Input/idmc_idus.xlsx` for trend analysis (also filtered for OND)
4. **Generate OND-specific visualizations** and save to `output/` folder

### Generated Visualizations:
| Output File | Description |
|-------------|-------------|
| `ond_displacements_by_country.png` | Bar chart of OND displacements by country (log scale) |
| `ond_displacement_choropleth_map.png` | Choropleth map of OND displacement figures |
| `ond_displacement_by_type_pie.png` | Pie chart of Disaster vs Conflict (OND) |
| `ond_monthly_displacement_by_type.png` | Stacked bar chart for Oct, Nov, Dec by type |
| `ond_displacements_by_disaster_type.png` | Bar chart by specific disaster type |
| `ond_disaster_type_pie.png` | Pie chart of disaster types |
| `ond_top_10_events.png` | Top 10 OND events causing displacement |
| `ond_trend_across_years.png` | OND trends compared across multiple years |
| `ond_total_by_year.png` | Total OND displacements by year (bar chart) |
| `ond_country_month_heatmap.png` | Heatmap of displacements by country and OND month |
| `ond_summary_statistics.txt` | Text file with OND summary statistics |

### Usage:
```bash
cd scripts
python displacement_analysis_ond.py
```

---

## Step 5: disaster_map_with_donuts_drm.py - DRM Impact Map

**Purpose:** Creates a disaster map with donut charts showing DRM Sector Impact Assessment data for OND 2025

### Workflow:
1. **Load** actual DRM Sector Impact Assessment data (hardcoded for OND 2025)
2. **Load** ICPAC shapefile for East Africa map
3. **Generate** map with disaster icons positioned on affected countries
4. **Create** three donut charts showing:
   - Total Events (top 2 most recurring hazards)
   - Total Deaths (top 2 deadliest hazards)
   - Total Affected (top 2 by affected population)
5. **Output:** `output/ond_disaster_map_with_donuts.png`

### Data Sources:
- IGAD-TAC, ECHO, IOM, UNOCHA, IPC, IFRC, WHO, ReliefWeb, FEWS NET

### Usage:
```bash
cd scripts
python disaster_map_with_donuts_drm.py
```

---

## Utility: change_color.py - Icon Recoloring

**Purpose:** Recolors SVG and PNG icons to a target color (default: green #088a4a)

### Features:
- Recolors SVG files by replacing fill/stroke attributes
- Converts SVGs to PNG for PDF compatibility (using cairosvg)
- Preserves transparency in PNG files

### Configuration:
```python
input_folder = "/path/to/original/icons"
output_folder = "/path/to/output/icons"
target_color_hex = "#088a4a"  # Green
```

### Usage:
```bash
cd scripts
python change_color.py
```

---

## Interactive Notebook: Displacement_Analysis_2025.ipynb

**Purpose:** Comprehensive interactive analysis of displacement data with multiple visualization types

### Input Data:
- `SummaryDATA_idmc_idus_May2025.xlsx` (184 records, 10 columns)
- `idmc_idus.xlsx` (for trend analysis)

### Analysis Sections:

#### A. Temporal Analysis
- Monthly displacement trends by year
- Line plots with seasonal patterns

#### B. Geographic Distribution
- Bar chart: Total displacements by country (log scale)
- Choropleth map: Displacement figures using ICPAC shapefile

#### C. Displacement Type Comparison
- Pie chart: Disaster vs Conflict proportions
- Stacked bar chart: Monthly breakdown by type

#### D. Duration-Based Insights
- Calculate `Duration_Days = end_date - start_date`
- Box plot: Duration distribution by displacement type

#### E. Event-Level Insights
- Top 10 events causing displacement (log-scale bar chart)

#### F. Long-term Trend Analysis
- Time series: 2018-2025 displacement trends
- Color-coded: Blue (pre-2025), Red (2025/OND season)

#### G. Map Visualizations with Donut Charts
- Eastern Africa map colored by disaster type
- Donut charts showing: Events, Deaths, Total Affected
- Icons for disaster types (Flood, Earthquake, Epidemic)

---

## Dependencies

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import numpy as np
import requests
import cairosvg  # For SVG to PNG conversion
from PIL import Image
from datetime import datetime
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
```

### Installation:
```bash
pip install pandas matplotlib seaborn plotly geopandas numpy requests cairosvg pillow openpyxl
```
