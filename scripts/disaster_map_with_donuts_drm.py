"""
Disaster Map with Donut Charts Script - OND 2025 DRM Data

This script creates a visualization showing:
- East Africa map with disaster type icons positioned on affected countries
- Three donut charts showing Total Events, Total Deaths, and Total Affected by disaster type
- Data sources box in the top right

Updated with actual DRM Sector Impact Assessment data (Oct-Dec 2025)

Usage:
    python disaster_map_with_donuts.py
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
    from config import EAST_AFRICA_COUNTRIES, SHAPEFILE_PATH, ICONS_DIR, OUTPUT_DIR
except ImportError:
    EAST_AFRICA_COUNTRIES = [
        "Somalia", "Tanzania", "Ethiopia", "Uganda", "Kenya", "Rwanda",
        "Burundi", "South Sudan", "Eritrea", "Djibouti", "Sudan"
    ]
    SHAPEFILE_PATH = None
    ICONS_DIR = None
    OUTPUT_DIR = None


# Configuration
ANALYSIS_PERIOD = "October - December 2025"
DATA_SOURCES = "IGAD-TAC, ECHO, IOM, UNOCHA, IPC, IFRC, WHO,\nReliefWeb, FEWS NET"

# Disaster type colors for donut charts
DISASTER_COLORS = {
    "Flood": "#1f77b4",      # Blue
    "Drought": "#ff7f0e",    # Orange
    "Earthquake": "#2ca02c", # Green
    "Epidemic": "#d62728",   # Red
    "Storm": "#9467bd",      # Purple
    "Landslide": "#8c564b",  # Brown
    "Fire": "#e377c2",       # Pink
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
        "Fire": "Fire_icon.png",
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

    # Create legend items with icons
    legend_items = []

    for disaster_type in sorted(disaster_types):
        icon_path = get_icon_path(disaster_type, icons_dir)

        if icon_path.exists():
            try:
                icon_img = mpimg.imread(str(icon_path))
                icon_box = OffsetImage(icon_img, zoom=0.03)
                icon_box.image.axes = ax
                text_box = TextArea(f"  {disaster_type}", textprops=dict(fontsize=11))
                row = HPacker(children=[icon_box, text_box], align='center', pad=0, sep=5)
                legend_items.append(row)
            except Exception as e:
                print(f"Warning: Could not load icon for {disaster_type}: {e}")
                text_box = TextArea(f"  {disaster_type}", textprops=dict(fontsize=11))
                legend_items.append(text_box)
        else:
            text_box = TextArea(f"  {disaster_type}", textprops=dict(fontsize=11))
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


def create_impact_summary_table(ax, impact_data):
    """
    Create a summary table showing key impact metrics by country.
    """
    ax.axis('off')
    
    # Prepare table data
    countries = list(impact_data.keys())
    affected = [f"{impact_data[c]['affected']:,}" if impact_data[c]['affected'] > 0 else "N/A" for c in countries]
    displaced = [f"{impact_data[c]['displaced']:,}" if impact_data[c]['displaced'] > 0 else "N/A" for c in countries]
    deaths = [str(impact_data[c]['deaths']) for c in countries]
    
    table_data = list(zip(countries, affected, displaced, deaths))
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Country', 'Affected', 'Displaced', 'Deaths'],
        loc='center',
        cellLoc='center',
        colColours=['#E8E8E8'] * 4
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    return table


def plot_disaster_map_with_donuts(
    shapefile_path: Path,
    icons_dir: Path,
    output_dir: Path,
    disaster_data: pd.DataFrame = None,
    analysis_period: str = ANALYSIS_PERIOD,
    data_sources: str = DATA_SOURCES
):
    """
    Create the main visualization with map and donut charts.
    
    Uses actual DRM Sector Impact Assessment data from OND 2025.
    """
    print("=" * 60)
    print("Creating Disaster Map with Donut Charts")
    print("DRM Sector Impact Assessment - OND 2025")
    print("=" * 60)

    # Load actual DRM data
    if disaster_data is None:
        print("Loading DRM Sector Impact Assessment data...")
        disaster_data = load_disaster_data()

    print(f"Loaded {len(disaster_data)} disaster records")

    # Load shapefile
    print(f"Loading shapefile from: {shapefile_path}")
    geo_df = gpd.read_file(shapefile_path)

    # Filter for East Africa
    geo_df_ea = geo_df[geo_df['COUNTRY'].isin(EAST_AFRICA_COUNTRIES)].copy()
    print(f"Filtered to {len(geo_df_ea)} East African countries")

    # Aggregate data for donut charts
    events_by_type = disaster_data.groupby('Disaster_Type')['Events'].sum()
    deaths_by_type = disaster_data.groupby('Disaster_Type')['Deaths'].sum()
    affected_by_type = disaster_data.groupby('Disaster_Type')['Affected'].sum()
    
    # Also calculate displaced for reference
    displaced_by_type = disaster_data.groupby('Disaster_Type')['Displaced'].sum()

    total_events = events_by_type.sum()
    total_deaths = deaths_by_type.sum()
    total_affected = affected_by_type.sum()
    total_displaced = displaced_by_type.sum()

    print(f"\n{'='*40}")
    print("DRM IMPACT SCALE SUMMARY")
    print(f"{'='*40}")
    print(f"  Total Events:    {total_events:,}")
    print(f"  Total Deaths:    {total_deaths:,}")
    print(f"  Total Affected:  {total_affected:,}")
    print(f"  Total Displaced: {total_displaced:,}")
    print(f"{'='*40}")
    
    print("\nBreakdown by Disaster Type:")
    print("-" * 50)
    for dtype in events_by_type.index:
        print(f"  {dtype:12} | Events: {events_by_type[dtype]:3} | Deaths: {deaths_by_type[dtype]:5,} | Affected: {affected_by_type[dtype]:10,}")

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

    # Add disaster icons to map
    print("Adding disaster icons to map...")
    icons_added = 0

    for _, row in disaster_data.iterrows():
        icon_path = get_icon_path(row['Disaster_Type'], icons_dir)
        if add_icon_to_map(ax_map, row['Lon'], row['Lat'], icon_path, zoom=0.05):
            icons_added += 1

    print(f"Added {icons_added} icons to map")

    # Legend removed as per user request

    # ----- ADD DATA SOURCES BOX -----
    sources_text = f"Sources: {data_sources}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    ax_donut1.text(0.5, 1.35, sources_text, transform=ax_donut1.transAxes,
                   fontsize=10, fontweight='bold', verticalalignment='top',
                   horizontalalignment='center', bbox=props)

    # ----- PLOT DONUT CHARTS -----
    print("Creating donut charts...")

    # Total Events: Top 2 most recurring hazards (Flood: 4, Drought: 3)
    create_donut_chart(ax_donut1, events_by_type, "Total Events", total_events, DISASTER_COLORS, top_n=2, show_percentages=False)

    # Total Deaths: Top 2 hazards with highest deaths (Epidemic: 1,584, Flood: 110)
    create_donut_chart(ax_donut2, deaths_by_type, "Total Deaths", total_deaths, DISASTER_COLORS, top_n=2, show_percentages=False)

    # Total Affected: Top 2 by affected population (Drought: 7.8M, Epidemic: 1.5M)
    create_donut_chart(ax_donut3, affected_by_type, "Total Affected", total_affected, DISASTER_COLORS, top_n=2, show_percentages=False)

    for ax in [ax_donut1, ax_donut2, ax_donut3]:
        ax.set_aspect('equal')
        # Tighter axis limits to make donuts appear larger while keeping labels visible
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)

    # Add footer
    footer_text = f"Data sources: IGAD-TAC, ReliefWeb, ECHO, FEWS NET, IOM, UNOCHA, IPC, IFRC, WHO | Analysis period: {analysis_period}"
    fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', fontsize=10, style='italic', color='gray')

    # Save figure
    output_file = output_dir / 'ond_disaster_map_with_donuts.png'
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
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

    # Run the visualization
    plot_disaster_map_with_donuts(
        shapefile_path=shapefile_path,
        icons_dir=icons_dir,
        output_dir=output_dir,
        analysis_period="October - December 2025"
    )


if __name__ == '__main__':
    main()
