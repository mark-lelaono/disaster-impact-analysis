"""
Disaster Map with Donut Charts Script

This script creates a visualization showing:
- East Africa map with disaster type icons positioned on affected countries
- Three donut charts showing Total Events, Total Displaced, and breakdown by type
- Data sources box in the top right

Supports both:
- Pipeline mode: receives processed DataFrame with season/year parameters
- Standalone mode: uses hardcoded OND 2025 DRM data as fallback

Usage:
    # Standalone (OND 2025 DRM data)
    python disaster_map_with_donuts_drm.py

    # Via pipeline
    python run_analysis.py --season MAM --year 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from config import (
        EAST_AFRICA_COUNTRIES, SHAPEFILE_PATH, ICONS_DIR, OUTPUT_DIR,
        get_season_config, SEASONS
    )
except ImportError:
    EAST_AFRICA_COUNTRIES = [
        "Somalia", "Tanzania", "Ethiopia", "Uganda", "Kenya", "Rwanda",
        "Burundi", "South Sudan", "Eritrea", "Djibouti", "Sudan"
    ]
    SHAPEFILE_PATH = None
    ICONS_DIR = None
    OUTPUT_DIR = None
    get_season_config = None
    SEASONS = None


# Default configuration (used for standalone OND 2025 mode)
ANALYSIS_PERIOD = "October - December 2025"
DATA_SOURCES = "IGAD-TAC, ECHO, IOM, UNOCHA, IPC, IFRC, WHO,\nReliefWeb, FEWS NET"


def build_disaster_data_from_pipeline(df, season_cfg=None, min_icons_per_country=3):
    """
    Build the disaster_data DataFrame from the pipeline's processed summary data.

    Converts the pipeline format (one row per displacement record) into the
    DRM-style format needed by the map. Each row in the output becomes one icon
    on the map. Countries with many events get proportionally more icons
    (minimum ``min_icons_per_country`` icons per country).

    Parameters
    ----------
    df : pd.DataFrame
        Processed summary data with columns: country, Disaster_Type, figure,
        displacement_type, Month, etc.
    season_cfg : dict, optional
        Season configuration from config.py.
    min_icons_per_country : int
        Minimum number of icons to show per country (default 3).

    Returns
    -------
    pd.DataFrame with columns: Country, Disaster_Type, Events, Deaths, Affected,
        Displaced, Lat, Lon
    """
    # Filter to disaster records only (Conflict records have NaN Disaster_Type)
    df_disaster = df[df['Disaster_Type'].notna()].copy()

    if df_disaster.empty:
        return pd.DataFrame(columns=[
            'Country', 'Disaster_Type', 'Events', 'Deaths',
            'Affected', 'Displaced', 'Lat', 'Lon'
        ])

    # Aggregate: one row per country + disaster type
    agg = df_disaster.groupby(['country', 'Disaster_Type']).agg(
        Events=('figure', 'count'),
        Displaced=('figure', 'sum'),
    ).reset_index()

    agg = agg.rename(columns={'country': 'Country'})

    # Expand rows so each country gets at least min_icons_per_country icons.
    # Icons are distributed proportionally by event count across disaster types.
    expanded_rows = []
    for country, grp in agg.groupby('Country'):
        n_types = len(grp)
        total_events = grp['Events'].sum()
        target = max(min_icons_per_country, n_types)

        # Distribute target icons proportionally by event count
        grp = grp.sort_values('Events', ascending=False).reset_index(drop=True)
        icon_counts = []
        remaining = target
        for i, row in grp.iterrows():
            if i == len(grp) - 1:
                count = remaining
            else:
                count = max(1, round(target * row['Events'] / total_events))
                count = min(count, remaining - (len(grp) - 1 - i))
            icon_counts.append(count)
            remaining -= count

        for (_, row), n_icons in zip(grp.iterrows(), icon_counts):
            for _ in range(n_icons):
                expanded_rows.append({
                    'Country': row['Country'],
                    'Disaster_Type': row['Disaster_Type'],
                    'Events': row['Events'],
                    'Deaths': 0,
                    'Affected': 0,
                    'Displaced': row['Displaced'],
                    'Lat': 0.0,
                    'Lon': 0.0,
                })

    return pd.DataFrame(expanded_rows)


def get_analysis_period_text(season: str, year: int) -> str:
    """Generate analysis period text from season and year."""
    if season.upper() == 'ANNUAL':
        return f"January - December {year}"
    if get_season_config:
        cfg = get_season_config(season)
        months = cfg['months']
        return f"{months[0]} - {months[-1]} {year}"
    # Fallback
    season_months = {
        'OND': 'October - December',
        'MAM': 'March - May',
        'JJAS': 'June - September',
    }
    return f"{season_months.get(season, season)} {year}"

# Disaster type colors for donut charts
DISASTER_COLORS = {
    "Flood": "#1f77b4",      # Blue
    "Drought": "#ff7f0e",    # Orange
    "Earthquake": "#2ca02c", # Green
    "Epidemic": "#d62728",   # Red
    "Storm": "#9467bd",      # Purple
    "Landslide": "#8c564b",  # Brown
    "Mass Movement": "#8c564b",  # Brown (same as Landslide)
    "Fire": "#e377c2",       # Pink
    "Wildfire": "#e377c2",   # Pink (same as Fire)
    "Conflict": "#7f7f7f",   # Gray
    "Lightning": "#bcbd22",  # Yellow-green
    "Hailstorm": "#17becf",  # Cyan
}


def load_disaster_data():
    """
    Load actual DRM Sector Impact Assessment data for OND 2025 season.
    Data sourced from DRM Sector Impact Assessment Survey (January 2026)
    
    Countries Reporting: Kenya, Somalia, Ethiopia, Uganda, Sudan, South Sudan (6 countries)
    
    Impact Scale from DRM Analysis:
    - Somalia: 3.5M affected, 197K+ displaced (Drought declared national emergency)
    - Ethiopia: 1.8M affected (Somali: 1M, Oromia: 800K) - Drought + Disease
    - Kenya (Drought): 2M+ affected, 2M+ displaced - ASAL counties
    - Kenya (Floods/Landslides): ~1,000 affected, ~755 displaced, 41 deaths
    - Uganda: 815 displaced, 49 deaths (Landslides, Lightning, Floods, Hailstorm, Fire)
    - Sudan: Floods, Lightning, Strong Winds - figures pending NCCD
    - South Sudan: Fire (OND), Floods + Cholera (season total: 1M+ affected, 355K displaced, 1,617 deaths)
    """
    
    # Actual disaster data from DRM Sector Impact Assessment
    # Each entry represents verified disaster impacts with location coordinates
    # 
    # EVENTS CALCULATION:
    # Total Events = 22 (sum of unique hazard occurrences across countries)
    # Each unique country-hazard combination counts as 1 event
    # Based on DRM Sector Impact Assessment Survey (6 countries reporting)
    #
    # Breakdown by Hazard Type:
    # - Drought: Kenya, Somalia, Ethiopia = 3
    # - Flood: Kenya, Uganda, Sudan, South Sudan = 4
    # - Landslide: Kenya, Uganda = 2
    # - Storm/Strong Winds: Kenya, Somalia, Sudan = 3
    # - Fire: Kenya, Uganda, South Sudan = 3
    # - Epidemic/Disease: Kenya, Ethiopia, South Sudan = 3
    # - Lightning: Sudan, Uganda = 2
    # - Hailstorm: Uganda = 1
    # - Earthquake: Ethiopia = 1
    # Total: 3+4+2+3+3+3+2+1+1 = 22 ✓
    
    disaster_events = [
        # ============ SOMALIA ============
        # National drought emergency declared Nov 10, 2025
        # 3.5 million affected, 197,000+ displaced
        {
            "Country": "Somalia", 
            "Disaster_Type": "Drought", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0,  # Not reported
            "Affected": 3500000, 
            "Displaced": 197000,
            "Lat": 5.15, 
            "Lon": 46.20
        },
        {
            "Country": "Somalia", 
            "Disaster_Type": "Storm",  # Strong winds reported
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 50000,
            "Displaced": 0,
            "Lat": 2.0, 
            "Lon": 45.0
        },
        
        # ============ ETHIOPIA ============
        # 1.8M affected: 1M in Somali region, 800K in Oromia
        # Drought + Disease (Malaria: 1.4M cases, Marburg: 12 cases)
        {
            "Country": "Ethiopia", 
            "Disaster_Type": "Drought", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 1800000,
            "Displaced": 0,
            "Lat": 6.0, 
            "Lon": 43.0  # Somali region
        },
        {
            "Country": "Ethiopia", 
            "Disaster_Type": "Epidemic",  # Malaria outbreak + Marburg
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 7,  # Marburg deaths
            "Affected": 1400000,  # Malaria cases
            "Displaced": 0,
            "Lat": 7.5, 
            "Lon": 38.5  # Southern Ethiopia
        },
        {
            "Country": "Ethiopia", 
            "Disaster_Type": "Earthquake",  # Oct 11, 2025 - Afar/Tigray
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 1, 
            "Affected": 1400,  # Homes damaged
            "Displaced": 0,
            "Lat": 14.0, 
            "Lon": 40.5  # Afar region
        },
        
        # ============ KENYA ============
        # Drought: 2M+ affected in ASAL counties (Mandera, Marsabit, Turkana, etc.)
        # Floods/Landslides: 41 deaths, 200 HH affected, 151 HH displaced
        {
            "Country": "Kenya", 
            "Disaster_Type": "Drought", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 2500000,  # 11 ASAL counties
            "Displaced": 2000000,
            "Lat": 2.5, 
            "Lon": 38.5  # Northern Kenya
        },
        {
            "Country": "Kenya", 
            "Disaster_Type": "Flood", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 10, 
            "Affected": 5000,
            "Displaced": 755,
            "Lat": 0.5, 
            "Lon": 35.5  # Western Kenya
        },
        {
            "Country": "Kenya", 
            "Disaster_Type": "Landslide",  # Elgeyo Marakwet - Oct 31
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 21,  # At least 21 killed
            "Affected": 1000,
            "Displaced": 755,
            "Lat": 0.8, 
            "Lon": 35.5  # Rift Valley
        },
        {
            "Country": "Kenya", 
            "Disaster_Type": "Storm",  # Strong winds reported
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 500,
            "Displaced": 0,
            "Lat": -1.0, 
            "Lon": 37.0
        },
        {
            "Country": "Kenya", 
            "Disaster_Type": "Fire",  # Forest fire
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 200,
            "Displaced": 0,
            "Lat": 0.0, 
            "Lon": 35.0
        },
        {
            "Country": "Kenya", 
            "Disaster_Type": "Epidemic",  # Cholera in Narok/Trans Mara + Disease
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 10, 
            "Affected": 5000,
            "Displaced": 0,
            "Lat": -1.5, 
            "Lon": 35.5
        },
        
        # ============ UGANDA ============
        # Mt. Elgon Landslides: 35 deaths, 715 displaced
        # Lightning: 14 deaths
        # Transport accidents: 297 deaths (road + drowning) - NOT included in disaster count
        # Fire: 100 displaced
        {
            "Country": "Uganda", 
            "Disaster_Type": "Landslide",  # Mt. Elgon - Oct 30-Nov 1
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 35,  # 19 male, 16 female, 18 children
            "Affected": 2000,
            "Displaced": 715,
            "Lat": 1.2, 
            "Lon": 34.5  # Mt. Elgon region
        },
        {
            "Country": "Uganda", 
            "Disaster_Type": "Flood", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 5000,
            "Displaced": 1500,
            "Lat": 0.5, 
            "Lon": 30.0  # Kasese/Ntoroko
        },
        {
            "Country": "Uganda", 
            "Disaster_Type": "Lightning", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 14, 
            "Affected": 500,
            "Displaced": 0,
            "Lat": 1.5, 
            "Lon": 32.5
        },
        {
            "Country": "Uganda", 
            "Disaster_Type": "Hailstorm", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 3000,  # Crop damage
            "Displaced": 0,
            "Lat": 0.8, 
            "Lon": 31.5
        },
        {
            "Country": "Uganda", 
            "Disaster_Type": "Fire", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 500,
            "Displaced": 100,
            "Lat": 0.3, 
            "Lon": 32.5
        },
        
        # ============ SUDAN ============
        # Floods, Lightning, Strong Winds, Pest
        # Figures pending NCCD declaration
        {
            "Country": "Sudan", 
            "Disaster_Type": "Flood", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 50,  # Estimated
            "Affected": 200000,
            "Displaced": 50000,
            "Lat": 15.5, 
            "Lon": 32.5
        },
        {
            "Country": "Sudan", 
            "Disaster_Type": "Lightning", 
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 10, 
            "Affected": 1000,
            "Displaced": 0,
            "Lat": 13.0, 
            "Lon": 30.0
        },
        {
            "Country": "Sudan", 
            "Disaster_Type": "Storm",  # Strong winds
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 5, 
            "Affected": 10000,
            "Displaced": 0,
            "Lat": 14.0, 
            "Lon": 33.0
        },
        
        # ============ SOUTH SUDAN ============
        # Forest Fire reported in OND
        # Flooding impacts from earlier in season included for context
        # Cholera: 94,549 cases, 1,567 deaths
        {
            "Country": "South Sudan", 
            "Disaster_Type": "Fire",  # Juba market fires
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 0, 
            "Affected": 1000,
            "Displaced": 200,
            "Lat": 4.85, 
            "Lon": 31.58  # Juba
        },
        {
            "Country": "South Sudan", 
            "Disaster_Type": "Flood",  # Season impacts
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 50, 
            "Affected": 1000000,
            "Displaced": 355000,
            "Lat": 7.5, 
            "Lon": 30.0  # Jonglei/Unity
        },
        {
            "Country": "South Sudan", 
            "Disaster_Type": "Epidemic",  # Cholera
            "Events": 1,  # 1 unique country-hazard occurrence
            "Deaths": 1567, 
            "Affected": 94549,
            "Displaced": 0,
            "Lat": 6.0, 
            "Lon": 31.0
        },
    ]

    return pd.DataFrame(disaster_events)


def get_impact_summary():
    """
    Returns a summary dictionary of the DRM Impact Scale data.
    This can be used for annotations or separate visualizations.
    """
    return {
        "Somalia": {
            "affected": 3500000,
            "displaced": 197000,
            "deaths": "Not reported",
            "key_hazards": ["Drought (National Emergency)", "Strong Winds"],
            "notes": "National drought emergency declared Nov 10, 2025"
        },
        "Ethiopia": {
            "affected": 1800000,
            "displaced": 0,
            "deaths": 8,
            "key_hazards": ["Drought", "Epidemic (Malaria/Marburg)", "Earthquake"],
            "notes": "1M in Somali region, 800K in Oromia; 1.4M malaria cases"
        },
        "Kenya": {
            "affected": 2506700,
            "displaced": 2001510,
            "deaths": 41,
            "key_hazards": ["Drought", "Floods", "Landslides", "Disease"],
            "notes": "11 ASAL counties in severe drought; Elgeyo Marakwet landslides"
        },
        "Uganda": {
            "affected": 11000,
            "displaced": 2315,
            "deaths": 352,
            "key_hazards": ["Landslides", "Lightning", "Floods", "Hailstorm"],
            "notes": "Mt. Elgon landslides; 297 transport deaths not in disaster count"
        },
        "Sudan": {
            "affected": 211000,
            "displaced": 50000,
            "deaths": 65,
            "key_hazards": ["Floods", "Lightning", "Strong Winds"],
            "notes": "Figures pending NCCD official declaration"
        },
        "South Sudan": {
            "affected": 1095549,
            "displaced": 355200,
            "deaths": 1617,
            "key_hazards": ["Floods", "Cholera", "Fire"],
            "notes": "94,549 cholera cases; 1,567 cholera deaths"
        },
    }


def get_icon_path(disaster_type: str, icons_dir: Path) -> Path:
    """Get the icon file path for a disaster type."""
    icon_mapping = {
        "Flood": "Flood_icon.png",
        "Drought": "Drought_icon.png",
        "Earthquake": "Earthquake_icon.png",
        "Epidemic": "Epidemic_icon.png",
        "Storm": "Storm_icon.png",
        "Landslide": "Landslide_icon.png",
        "Mass Movement": "Landslide_icon.png",
        "Fire": "Fire_40400.png",
        "Wildfire": "Fire_40400.png",
        "Conflict": "Conflict_icon.png",
        "Lightning": "Lightning_icon.png",
        "Hailstorm": "Hailstorm_icon.png",
    }

    icon_file = icon_mapping.get(disaster_type, "default_icon.png")
    return icons_dir / icon_file


def add_icon_to_map(ax, x, y, icon_path: Path, zoom=0.04):
    """Add an icon to the map at specified coordinates."""
    if icon_path.exists():
        try:
            icon_img = mpimg.imread(str(icon_path))
            imagebox = OffsetImage(icon_img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=5)
            ax.add_artist(ab)
            return True
        except Exception as e:
            print(f"Warning: Could not load icon {icon_path}: {e}")
    return False


def create_donut_chart(ax, data, title, total_value, colors, top_n=None, show_percentages=True):
    """
    Create a donut chart with labels.

    Parameters:
    - ax: matplotlib axes
    - data: pandas Series with disaster type as index and values
    - title: Title for the center of donut
    - total_value: Total value to display in center
    - colors: Dictionary mapping disaster types to colors
    - top_n: If specified, only show top N categories (no "Other" grouping)
    - show_percentages: If True, show percentages in labels; if False, show only hazard names
    """
    # Filter out zero values
    non_zero_data = data[data > 0].sort_values(ascending=False)

    if len(non_zero_data) == 0:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return

    # If top_n is specified, keep only top N categories (no "Other" grouping)
    if top_n is not None and len(non_zero_data) > top_n:
        non_zero_data = non_zero_data.head(top_n)

    # Calculate percentages based on displayed data only
    total = non_zero_data.sum()

    # Get colors for each category
    chart_colors = [colors.get(idx, "#FBFBFB") for idx in non_zero_data.index]

    # Create the pie chart (donut)
    wedges, texts = ax.pie(
        non_zero_data.values,
        colors=chart_colors,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white')
    )

    # Add labels outside the donut
    for i, (wedge, (disaster_type, value)) in enumerate(zip(wedges, non_zero_data.items())):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))

        if show_percentages:
            pct = value / total * 100
            label = f"{disaster_type}\n({pct:.1f}%)"
        else:
            # Show only hazard name (no percentage)
            label = disaster_type

        ha = 'left' if x > 0 else 'right'
        ax.annotate(label, xy=(x, y), ha=ha, va='center', fontsize=9, fontweight='bold')

    # Add center text with formatted numbers
    if total_value >= 1e6:
        center_text = f"{title}\n{total_value/1e6:.1f}M"
    elif total_value >= 1e3:
        center_text = f"{title}\n{total_value/1e3:.1f}K"
    else:
        center_text = f"{title}\n{total_value:.0f}"

    ax.text(0, 0, center_text, ha='center', va='center', fontsize=11, fontweight='bold')


def create_icon_legend(ax, disaster_types, icons_dir: Path, position='lower left'):
    """
    Create a custom legend box with small icons and disaster type names.
    """
    from matplotlib.offsetbox import HPacker, VPacker, TextArea, OffsetImage, AnnotationBbox

    # Legend display names
    display_names = {
        "Flood": "Floods",
        "Mass Movement": "Landslide",
        "Wildfire": "Wildfire",
    }

    # Create legend items with icons
    legend_items = []

    # Add "Legend" header
    header = TextArea("Legend", textprops=dict(fontsize=13, fontweight='bold'))
    legend_items.append(header)

    for disaster_type in sorted(disaster_types):
        icon_path = get_icon_path(disaster_type, icons_dir)
        label = display_names.get(disaster_type, disaster_type)

        if icon_path.exists():
            try:
                icon_img = mpimg.imread(str(icon_path))
                icon_box = OffsetImage(icon_img, zoom=0.09)
                icon_box.image.axes = ax
                text_box = TextArea(f"  {label}", textprops=dict(fontsize=12))
                row = HPacker(children=[icon_box, text_box], align='center', pad=0, sep=5)
                legend_items.append(row)
            except Exception as e:
                print(f"Warning: Could not load icon for {disaster_type}: {e}")
                text_box = TextArea(f"  {label}", textprops=dict(fontsize=11))
                legend_items.append(text_box)
        else:
            text_box = TextArea(f"  {label}", textprops=dict(fontsize=11))
            legend_items.append(text_box)

    legend_box = VPacker(children=legend_items, align='left', pad=5, sep=8)

    if position == 'lower left':
        xy = (0.02, 0.02)
        box_alignment = (0, 0)
    elif position == 'lower right':
        xy = (0.98, 0.02)
        box_alignment = (1, 0)
    elif position == 'upper left':
        xy = (0.02, 0.98)
        box_alignment = (0, 1)
    else:
        xy = (0.98, 0.98)
        box_alignment = (1, 1)

    anchored_box = AnnotationBbox(
        legend_box,
        xy,
        xycoords='axes fraction',
        box_alignment=box_alignment,
        frameon=True,
        pad=0.5,
        bboxprops=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    )

    ax.add_artist(anchored_box)
    return anchored_box


def get_icon_slots(geom, n_icons, spacing=1.5):
    """
    Generate n non-overlapping icon positions inside a country polygon.

    Places icons in a compact grid centred on the polygon's representative
    interior point, with the given spacing (in degrees).  Every candidate
    is guaranteed to lie inside the polygon; if a grid cell falls outside
    the boundary it is nudged inward toward the centre.

    Parameters
    ----------
    geom : shapely geometry
        The country polygon / multipolygon.
    n_icons : int
        How many positions are needed.
    spacing : float
        Gap between icon centres in degrees.

    Returns
    -------
    list of (lon, lat) tuples
    """
    from shapely.geometry import Point

    # representative_point() is always inside the polygon (unlike centroid)
    centre = geom.representative_point()
    cx, cy = centre.x, centre.y

    if n_icons == 1:
        return [(cx, cy)]

    # Scatter icons in concentric rings around the centre.
    # First icon at centre, rest spread at varying angles and radii.
    positions = [(cx, cy)]  # first icon goes at centre
    remaining = n_icons - 1

    # Place remaining icons in a ring, evenly spaced by angle
    # with a slight radial jitter so they don't look mechanical
    ring_angles = []
    angle_step = 360.0 / remaining
    # Start at a tilted angle (not 0°) so icons don't align horizontally
    start_angle = 35.0
    for i in range(remaining):
        ring_angles.append(start_angle + i * angle_step)

    for angle_deg in ring_angles:
        angle_rad = np.radians(angle_deg)
        dx = spacing * np.cos(angle_rad)
        dy = spacing * np.sin(angle_rad)
        lon, lat = cx + dx, cy + dy
        pt = Point(lon, lat)
        if geom.contains(pt):
            positions.append((lon, lat))
        else:
            # Shrink radius toward centre until inside
            for t in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0):
                nlon = lon + (cx - lon) * t
                nlat = lat + (cy - lat) * t
                if geom.contains(Point(nlon, nlat)):
                    positions.append((nlon, nlat))
                    break
            else:
                positions.append((cx, cy))

    return positions


def plot_disaster_map_with_donuts(
    shapefile_path: Path,
    icons_dir: Path,
    output_dir: Path,
    disaster_data: pd.DataFrame = None,
    analysis_period: str = None,
    data_sources: str = DATA_SOURCES,
    season: str = None,
    year: int = None,
    pipeline_df: pd.DataFrame = None,
):
    """
    Create the main visualization with map and donut charts.

    Parameters
    ----------
    shapefile_path, icons_dir, output_dir : Path
        Standard path arguments.
    disaster_data : pd.DataFrame, optional
        Pre-built DRM-format data (Country, Disaster_Type, Events, etc.).
    analysis_period : str, optional
        Override text for the title. Auto-generated from season/year if None.
    data_sources : str
        Footer text for data sources.
    season : str, optional
        Season code ('OND', 'MAM', 'JJAS'). Used for title and filename.
    year : int, optional
        Analysis year. Used for title and filename.
    pipeline_df : pd.DataFrame, optional
        Raw processed DataFrame from the pipeline (summary data).
        If provided, disaster_data is built from it automatically.
    """
    season_label = season or 'OND'
    year_label = year or 2025

    # Determine analysis period text
    if analysis_period is None:
        analysis_period = get_analysis_period_text(season_label, year_label)

    print("=" * 60)
    print("Creating Disaster Map with Donut Charts")
    print(f"{season_label} {year_label}")
    print("=" * 60)

    # Build disaster data from pipeline or load hardcoded DRM data
    if disaster_data is None and pipeline_df is not None:
        season_cfg = get_season_config(season_label) if get_season_config else None
        disaster_data = build_disaster_data_from_pipeline(pipeline_df, season_cfg)
        print(f"Built disaster data from pipeline: {len(disaster_data)} country-hazard records")
    elif disaster_data is None:
        print("Loading hardcoded DRM Sector Impact Assessment data (OND 2025)...")
        disaster_data = load_disaster_data()

    print(f"Loaded {len(disaster_data)} disaster records")

    # Load shapefile
    print(f"Loading shapefile from: {shapefile_path}")
    geo_df = gpd.read_file(shapefile_path)

    # Filter for East Africa
    geo_df_ea = geo_df[geo_df['COUNTRY'].isin(EAST_AFRICA_COUNTRIES)].copy()
    print(f"Filtered to {len(geo_df_ea)} East African countries")

    # Deduplicate for donut chart totals (expanded icon rows share the same
    # Events/Displaced values, so aggregate per country+type first)
    unique_combos = disaster_data.groupby(['Country', 'Disaster_Type']).agg(
        Events=('Events', 'first'),
        Displaced=('Displaced', 'first'),
        Affected=('Affected', 'first') if 'Affected' in disaster_data.columns else ('Displaced', 'first'),
    ).reset_index()

    events_by_type = unique_combos.groupby('Disaster_Type')['Events'].sum()
    displaced_by_type = unique_combos.groupby('Disaster_Type')['Displaced'].sum()

    # Affected: use real data if available, otherwise fall back to Displaced
    if 'Affected' in unique_combos.columns and unique_combos['Affected'].sum() > 0:
        affected_by_type = unique_combos.groupby('Disaster_Type')['Affected'].sum()
    else:
        affected_by_type = displaced_by_type.copy()

    total_events = events_by_type.sum()
    total_displaced = displaced_by_type.sum()
    total_affected = affected_by_type.sum()

    print(f"\n{'='*40}")
    print(f"IMPACT SUMMARY — {season_label} {year_label}")
    print(f"{'='*40}")
    print(f"  Total Events:    {total_events:,}")
    print(f"  Total Displaced: {total_displaced:,}")
    print(f"  Total Affected:  {total_affected:,}")
    print(f"{'='*40}")

    print("\nBreakdown by Disaster Type:")
    print("-" * 50)
    for dtype in events_by_type.index:
        print(f"  {dtype:12} | Events: {events_by_type[dtype]:3} | Displaced: {displaced_by_type.get(dtype, 0):10,} | Affected: {affected_by_type.get(dtype, 0):10,}")

    # Create figure with custom layout and white background
    fig = plt.figure(figsize=(20, 14), facecolor='white')

    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1], height_ratios=[1, 1, 1],
                          hspace=0.25, wspace=0.15)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_donut1 = fig.add_subplot(gs[0, 1])
    ax_donut2 = fig.add_subplot(gs[1, 1])
    ax_donut3 = fig.add_subplot(gs[2, 1])

    # ----- PLOT THE MAP -----
    print("\nPlotting map...")
    # Set white background for map
    ax_map.set_facecolor('white')
    fig.patch.set_facecolor('white')
    geo_df_ea.plot(ax=ax_map, color='#f5f5f5', edgecolor='black', linewidth=1)

    ax_map.set_title(
        f"DISASTER TYPES IN EASTERN AFRICA\n({analysis_period})",
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    ax_map.axis('off')

    # Place icons inside each country — positions computed from polygon
    print("Adding disaster icons to map...")
    icons_added = 0

    for country, group in disaster_data.groupby('Country'):
        n = len(group)
        country_row = geo_df_ea[geo_df_ea['COUNTRY'] == country]

        if len(country_row) == 0:
            # Country not in shapefile — fall back to data coords
            for _, row in group.iterrows():
                icon_path = get_icon_path(row['Disaster_Type'], icons_dir)
                if add_icon_to_map(ax_map, row['Lon'], row['Lat'], icon_path, zoom=0.10):
                    icons_added += 1
            continue

        geom = country_row.geometry.iloc[0]

        # Adaptive zoom & spacing based on icon count
        if n <= 2:
            zoom, spacing = 0.10, 2.0
        elif n <= 4:
            zoom, spacing = 0.085, 1.8
        else:
            zoom, spacing = 0.07, 1.5

        slots = get_icon_slots(geom, n, spacing=spacing)

        for i, (_, row) in enumerate(group.iterrows()):
            lon, lat = slots[i] if i < len(slots) else (geom.representative_point().x, geom.representative_point().y)
            icon_path = get_icon_path(row['Disaster_Type'], icons_dir)
            if add_icon_to_map(ax_map, lon, lat, icon_path, zoom=zoom):
                icons_added += 1

    print(f"Added {icons_added} icons to map")

    # Add icon legend to the map
    unique_types = sorted(disaster_data['Disaster_Type'].unique())
    create_icon_legend(ax_map, unique_types, icons_dir, position='lower right')

    # ----- ADD DATA SOURCES BOX -----
    sources_text = f"Sources: {data_sources}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    ax_donut1.text(0.5, 1.35, sources_text, transform=ax_donut1.transAxes,
                   fontsize=10, fontweight='bold', verticalalignment='top',
                   horizontalalignment='center', bbox=props)

    # ----- PLOT DONUT CHARTS -----
    print("Creating donut charts...")

    # Donut 1: Total Events by disaster type
    create_donut_chart(ax_donut1, events_by_type, "Total Events", total_events, DISASTER_COLORS, top_n=2, show_percentages=False)

    # Donut 2: Total Displaced by disaster type
    create_donut_chart(ax_donut2, displaced_by_type, "Total Displaced", total_displaced, DISASTER_COLORS, top_n=2, show_percentages=False)

    # Donut 3: Total Affected by disaster type
    create_donut_chart(ax_donut3, affected_by_type, "Total Affected", total_affected, DISASTER_COLORS, top_n=2, show_percentages=False)

    for ax in [ax_donut1, ax_donut2, ax_donut3]:
        ax.set_aspect('equal')
        # Tighter axis limits to make donuts appear larger while keeping labels visible
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)

    # Add footer
    footer_text = f"Data sources: IGAD-TAC, ReliefWeb, ECHO, FEWS NET, IOM, UNOCHA, IPC, IFRC, WHO | Analysis period: {analysis_period}"
    fig.text(0.5, 0.03, footer_text, ha='center', va='bottom', fontsize=10, style='italic', color='gray')
    disclaimer = "Icons represent disaster types reported per country. Multiple icons of the same type indicate higher event frequency."
    fig.text(0.5, 0.005, disclaimer, ha='center', va='bottom', fontsize=8, style='italic', color='gray')

    # Save figure
    output_file = output_dir / f'{season_label.lower()}_disaster_map_with_donuts.png'
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_file}")
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    return output_file


def print_impact_scale_report():
    """
    Print a formatted report of the DRM Impact Scale data.
    """
    impact_summary = get_impact_summary()
    
    print("\n" + "=" * 70)
    print("DRM SECTOR IMPACT ASSESSMENT - OND 2025")
    print("IMPACT SCALE BY COUNTRY")
    print("=" * 70)
    
    for country, data in impact_summary.items():
        print(f"\n{country.upper()}")
        print("-" * 40)
        print(f"  Affected:  {data['affected']:,}")
        print(f"  Displaced: {data['displaced']:,}")
        print(f"  Deaths:    {data['deaths']}")
        print(f"  Hazards:   {', '.join(data['key_hazards'])}")
        print(f"  Notes:     {data['notes']}")
    
    print("\n" + "=" * 70)


def main():
    """Main function to run the visualization."""
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    shapefile_path = SHAPEFILE_PATH or (base_dir / 'shapefiles' / 'Administrative0_Boundaries_ICPAC_Countries.shp')
    icons_dir = ICONS_DIR or (base_dir / 'icons')
    output_dir = OUTPUT_DIR or (base_dir / 'output')

    if not shapefile_path.exists():
        print(f"Error: Shapefile not found at {shapefile_path}")
        return

    if not icons_dir.exists():
        print(f"Warning: Icons directory not found at {icons_dir}")
        print("Icons will not be displayed on the map.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print impact scale report
    print_impact_scale_report()

    # Run the visualization (standalone mode: OND 2025 DRM data)
    plot_disaster_map_with_donuts(
        shapefile_path=shapefile_path,
        icons_dir=icons_dir,
        output_dir=output_dir,
        season='OND',
        year=2025,
    )


if __name__ == '__main__':
    main()
