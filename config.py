"""
Central configuration for the Disaster Displacement Analysis pipeline.

All shared constants, paths, season definitions, and database settings.
"""

from pathlib import Path

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent

INPUT_DIR = PROJECT_ROOT / 'Input'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SHAPEFILES_DIR = PROJECT_ROOT / 'shapefiles'
ICONS_DIR = PROJECT_ROOT / 'icons'
DATA_DIR = PROJECT_ROOT / 'data'

SHAPEFILE_PATH = SHAPEFILES_DIR / 'Administrative0_Boundaries_ICPAC_Countries.shp'
RAW_DATA_FILENAME = 'idmc_idus.xlsx'
SUMMARY_DATA_PATTERN = '*_SummaryDATA_idmc_idus.xlsx'

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_PATH = DATA_DIR / 'analysis.db'

# ============================================================
# API CONFIGURATION
# ============================================================

IDMC_API_URL = "https://helix-tools-api.idmcdb.org/external-api/idus/all/"
IDMC_CLIENT_ID = "ICPACAUG10"

# ============================================================
# COUNTRY DEFINITIONS
# ============================================================

EAST_AFRICA_COUNTRIES = [
    "Somalia", "Tanzania", "Ethiopia", "Uganda", "Kenya",
    "Rwanda", "Burundi", "South Sudan", "Eritrea", "Djibouti", "Sudan"
]

EAST_AFRICA_ISO3 = [
    'RWA', 'UGA', 'TZA', 'BDI', 'SSD', 'SDN',
    'ERI', 'ETH', 'DJI', 'SOM', 'KEN'
]

# ============================================================
# SEASON DEFINITIONS
# ============================================================

SEASONS = {
    'OND': {
        'name': 'OND',
        'long_name': 'October-November-December',
        'months': ['October', 'November', 'December'],
        'month_numbers': [10, 11, 12],
    },
    'JJAS': {
        'name': 'JJAS',
        'long_name': 'June-July-August-September',
        'months': ['June', 'July', 'August', 'September'],
        'month_numbers': [6, 7, 8, 9],
    },
    'MAM': {
        'name': 'MAM',
        'long_name': 'March-April-May',
        'months': ['March', 'April', 'May'],
        'month_numbers': [3, 4, 5],
    },
}

ALL_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# ============================================================
# VISUALIZATION DEFAULTS
# ============================================================

DATA_SOURCE_ORGANIZATIONS = "ECHO, FEWS NET, IOM, UNOCHA, IPC, IFRC, WHO"


def get_season_config(season_name: str) -> dict:
    """Get season configuration, raising ValueError for invalid seasons."""
    season_name = season_name.upper()
    if season_name not in SEASONS:
        valid = ', '.join(SEASONS.keys())
        raise ValueError(f"Unknown season '{season_name}'. Valid seasons: {valid}")
    return SEASONS[season_name]


def get_data_source_footer(season_name: str, year: int) -> str:
    """Generate the data source footer text for a given season and year."""
    season = get_season_config(season_name)
    month_range = f"{season['months'][0]}\u2013{season['months'][-1]}"
    return (
        f"Data sources: {DATA_SOURCE_ORGANIZATIONS} | "
        f"Analysis period: {month_range} {year}"
    )
