"""
Displacement Analysis Script

This script performs comprehensive analysis of IDMC displacement data
and generates visualizations. It reads the summary data from the output folder
and saves all visualizations there as well.

Usage:
    python displacement_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import glob


def format_number(value, pos):
    """Format numbers with K for thousands and M for millions."""
    if value >= 1e6:
        return f'{value/1e6:.1f}M'
    elif value >= 1e3:
        return f'{value/1e3:.0f}K'
    else:
        return f'{value:.0f}'


def get_latest_summary_file(output_dir: Path) -> Path:
    """Find the most recent summary file in the output directory."""
    pattern = str(output_dir / '*_SummaryDATA_idmc_idus.xlsx')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No summary files found matching pattern: {pattern}")
    # Return the most recently modified file
    return Path(max(files, key=lambda x: Path(x).stat().st_mtime))


def load_data(summary_file: Path, input_dir: Path):
    """Load summary and raw data files."""
    print(f"Loading summary data from: {summary_file}")
    df = pd.read_excel(summary_file)
    print(f"Summary data loaded: {len(df)} records")

    # Load raw data for trend analysis
    raw_file = input_dir / 'idmc_idus.xlsx'
    print(f"Loading raw data from: {raw_file}")
    dff = pd.read_excel(raw_file)
    print(f"Raw data loaded: {len(dff)} records")

    return df, dff


def plot_geographic_distribution(df: pd.DataFrame, shapefile_path: Path, output_dir: Path):
    """Plot geographic distribution of displacements."""
    print("\n--- Geographic Distribution ---")

    # Displacements by country (log-scaled bar plot)
    country_figures = df.groupby('country')['figure'].sum().reset_index().sort_values(by='figure', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=country_figures, y='country', x='figure', hue='country', palette='tab10')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_number))
    plt.title("Total Displacements by Country", fontsize=16)
    plt.xlabel("Displacements", fontsize=14)
    plt.ylabel("Country", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'displacements_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: displacements_by_country.png")

    # Choropleth map
    geo_df = gpd.read_file(shapefile_path)
    country_figures_map = df.groupby("country")["figure"].sum().reset_index()
    merged = geo_df.merge(country_figures_map, how='left', left_on='COUNTRY', right_on='country')
    merged['figure'] = merged['figure'].fillna(0)
    merged['log_figure'] = np.log1p(merged['figure'])

    fig, ax = plt.subplots(figsize=(14, 8))
    merged.plot(column='log_figure', cmap='OrRd', legend=True, edgecolor='black', ax=ax)
    plt.title('Total Displacement Figures by Country', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_choropleth_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: displacement_choropleth_map.png")


def plot_displacement_type_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot displacement type comparison charts."""
    print("\n--- Displacement Type Comparison ---")

    # Pie Chart
    type_summary = df.groupby('displacement_type')['figure'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.pie(
        type_summary['figure'],
        labels=type_summary['displacement_type'],
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 12}
    )
    plt.title("Displacement by Type", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_by_type_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: displacement_by_type_pie.png")

    # Stacked Bar Chart
    # Define month order from January to December
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    type_month = df.groupby(['Month', 'displacement_type'])['figure'].sum().unstack().fillna(0)

    # Reindex to order months from January to December (only include months that exist in data)
    available_months = [m for m in month_order if m in type_month.index]
    type_month = type_month.reindex(available_months)

    ax = type_month.plot(kind='bar', stacked=True, figsize=(12, 6))

    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    plt.title("Monthly Displacement by Type", fontsize=16)
    plt.ylabel("Displacements", fontsize=14)
    plt.xlabel("Month", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Displacement Type", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_displacement_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: monthly_displacement_by_type.png")


def plot_top_events(df: pd.DataFrame, output_dir: Path):
    """Plot top 10 events causing displacement."""
    print("\n--- Event-Level Insights ---")

    top_events = df.groupby('event_name')['figure'].sum().sort_values(ascending=False).head(10).reset_index()

    plt.figure(figsize=(20, 10))
    sns.barplot(data=top_events, y='event_name', hue='event_name', x='figure', palette='tab10')
    plt.xscale('log')
    plt.title("Top 10 Events Causing Displacement", fontsize=30)
    plt.xlabel("Displacements", fontsize=30)
    plt.ylabel("Event Name", fontsize=30)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_10_events.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: top_10_events.png")


def plot_displacement_trend(dff: pd.DataFrame, output_dir: Path):
    """Plot displacement trends over time."""
    print("\n--- Displacement Trends Over Time ---")

    # Ensure datetime format
    dff['displacement_start_date'] = pd.to_datetime(dff['displacement_start_date'])

    # Create Year-Month time series and Year/Month columns
    dff['YearMonth'] = dff['displacement_start_date'].dt.to_period('M').dt.to_timestamp()
    dff['Year'] = dff['displacement_start_date'].dt.year
    dff['MonthNum'] = dff['displacement_start_date'].dt.month

    # Group displacement figures by Year-Month
    monthly_trend = dff.groupby('YearMonth')['figure'].sum().reset_index()
    monthly_trend = monthly_trend[monthly_trend['figure'] > 0]
    monthly_trend = monthly_trend.merge(
        dff[['YearMonth', 'Year', 'MonthNum']].drop_duplicates(),
        on='YearMonth',
        how='left'
    )

    # Plot with log scale
    # Blue: Pre-2025 and Jan-Sep 2025
    # Red: Oct-Dec 2025 (OND season)
    plt.figure(figsize=(14, 6))

    # Sort by YearMonth
    monthly_trend = monthly_trend.sort_values('YearMonth')

    # Pre-2025 and Jan-Sep 2025 data (blue)
    data_blue = monthly_trend[
        (monthly_trend['Year'] < 2025) |
        ((monthly_trend['Year'] == 2025) & (monthly_trend['MonthNum'] < 10))
    ]
    if not data_blue.empty:
        sns.lineplot(data=data_blue, x='YearMonth', y='figure', marker='o', color='blue',
                    label='Pre-OND 2025', linewidth=2, markersize=6)

    # Oct-Dec 2025 data (red) - OND season
    data_2025_ond = monthly_trend[
        (monthly_trend['Year'] == 2025) & (monthly_trend['MonthNum'] >= 10)
    ]
    if not data_2025_ond.empty:
        sns.lineplot(data=data_2025_ond, x='YearMonth', y='figure', marker='o', color='red',
                    label='OND-2025', linewidth=2, markersize=6)

    # Connect the last blue point to the first red point with a continuous line
    if not data_blue.empty and not data_2025_ond.empty:
        last_blue = data_blue.iloc[-1]
        first_red = data_2025_ond.iloc[0]
        plt.plot([last_blue['YearMonth'], first_red['YearMonth']],
                [last_blue['figure'], first_red['figure']],
                color='blue', linestyle='-', linewidth=2)

    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))
    plt.title("Displacement Trend Over Time in Eastern Africa", fontsize=16)
    plt.xlabel("Years", fontsize=16)
    plt.ylabel("Number of Displacements", fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(fontsize=16, title='Period', title_fontsize='13')
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_trend_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: displacement_trend_over_time.png")


def plot_displacement_trend_2025(dff: pd.DataFrame, output_dir: Path):
    """Plot displacement trends for 2025 only (Jan-Sep blue, OND red)."""
    print("\n--- Displacement Trends 2025 Only ---")

    # Ensure datetime format
    dff['displacement_start_date'] = pd.to_datetime(dff['displacement_start_date'])

    # Create Year-Month time series and Year/Month columns
    dff['YearMonth'] = dff['displacement_start_date'].dt.to_period('M').dt.to_timestamp()
    dff['Year'] = dff['displacement_start_date'].dt.year
    dff['MonthNum'] = dff['displacement_start_date'].dt.month
    dff['MonthName'] = dff['displacement_start_date'].dt.strftime('%B')

    # Filter for 2025 only
    dff_2025 = dff[dff['Year'] == 2025].copy()

    # Group displacement figures by Year-Month
    monthly_trend = dff_2025.groupby('YearMonth')['figure'].sum().reset_index()
    monthly_trend = monthly_trend[monthly_trend['figure'] > 0]
    monthly_trend = monthly_trend.merge(
        dff_2025[['YearMonth', 'Year', 'MonthNum', 'MonthName']].drop_duplicates(),
        on='YearMonth',
        how='left'
    )

    # Sort by month
    monthly_trend = monthly_trend.sort_values('MonthNum')

    # Plot - 2025 only
    # Blue: Jan-Sep, Red: Oct-Dec (OND season)
    plt.figure(figsize=(14, 6))

    # Jan-Sep 2025 data (blue)
    data_jan_sep = monthly_trend[monthly_trend['MonthNum'] < 10]
    if not data_jan_sep.empty:
        sns.lineplot(data=data_jan_sep, x='MonthName', y='figure', marker='o', color='blue', label='Jan-Sep 2025', linewidth=2, markersize=8)

    # Oct-Dec 2025 data (red) - OND season
    data_ond = monthly_trend[monthly_trend['MonthNum'] >= 10]
    if not data_ond.empty:
        sns.lineplot(data=data_ond, x='MonthName', y='figure', marker='o', color='red', label='OND 2025', linewidth=2, markersize=8)

    # Connect the last blue point to the first red point with a continuous line
    if not data_jan_sep.empty and not data_ond.empty:
        last_jan_sep = data_jan_sep.iloc[-1]
        first_ond = data_ond.iloc[0]
        plt.plot([last_jan_sep['MonthName'], first_ond['MonthName']],
                [last_jan_sep['figure'], first_ond['figure']],
                color='blue', linestyle='-', linewidth=2)

    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))
    plt.title("Displacement Trend in Eastern Africa - 2025", fontsize=16)
    plt.xlabel("2025 Months", fontsize=16)
    plt.ylabel("Number of Displacements", fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(fontsize=14, title='Period', title_fontsize='13')
    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_trend_2025_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: displacement_trend_2025_only.png")


def plot_disaster_map_with_donuts(shapefile_path: Path, icons_dir: Path, output_dir: Path):
    """Plot Eastern Africa disaster map with donut charts."""
    print("\n--- Disaster Map with Donut Charts ---")

    # Prepare the disaster data (sample data - can be updated with actual data)
    data = [
        {"Country": "Somalia", "Disaster Type": "Floods", "Events": 1, "Deaths": 4, "Total Affected": 30000},
        {"Country": "Tanzania", "Disaster Type": "Floods", "Events": 1, "Deaths": 0, "Total Affected": 2165},
        {"Country": "Ethiopia", "Disaster Type": "Earthquake", "Events": 1, "Deaths": 0, "Total Affected": 80000},
        {"Country": "Uganda", "Disaster Type": "Epidemic", "Events": 1, "Deaths": 4, "Total Affected": 0}
    ]

    df_disaster = pd.DataFrame(data)
    geo_df = gpd.read_file(shapefile_path)

    east_africa_countries = ["Somalia", "Tanzania", "Ethiopia", "Uganda", "Kenya", "Rwanda",
                            "Burundi", "South Sudan", "Eritrea", "Djibouti", "Sudan"]
    geo_df_east_africa = geo_df[geo_df['COUNTRY'].isin(east_africa_countries)]

    merged = geo_df_east_africa.merge(df_disaster, how='left', left_on='COUNTRY', right_on='Country')

    disaster_colors = {
        "Floods": "blue",
        "Earthquake": "orange",
        "Epidemic": "purple",
        np.nan: "lightgray"
    }
    merged['Color'] = merged['Disaster Type'].map(disaster_colors).fillna("lightgray")
    merged['coords'] = merged['geometry'].representative_point().apply(lambda p: (p.x, p.y))

    # Icon paths
    icon_paths = {
        "Floods": icons_dir / "Flood_icon.png",
        "Earthquake": icons_dir / "Earthquake_icon.png",
        "Epidemic": icons_dir / "Epidemic_icon.png"
    }

    def add_icon(ax, x, y, icon_path, zoom=0.5):
        if icon_path.exists():
            icon_img = mpimg.imread(str(icon_path))
            imagebox = OffsetImage(icon_img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
    ax_map = fig.add_subplot(gs[:, 0])

    merged.plot(ax=ax_map, color=merged['Color'], edgecolor='black')
    ax_map.set_title("Disaster Types in East Africa (2025)", fontsize=16)
    ax_map.axis('off')

    # Add disaster icons
    for _, row in merged.iterrows():
        disaster_type = row['Disaster Type']
        if pd.notna(disaster_type) and disaster_type in icon_paths:
            x, y = row['coords']
            add_icon(ax_map, x, y, icon_paths[disaster_type], zoom=0.08)

    # Add legend
    legend_elements = [
        Patch(facecolor="blue", label="Floods"),
        Patch(facecolor="orange", label="Earthquake"),
        Patch(facecolor="purple", label="Epidemic"),
        Patch(facecolor="lightgray", label="No Disaster Reported")
    ]
    ax_map.legend(handles=legend_elements, title="Disaster Types", loc='lower left', fontsize=13, title_fontsize=14)

    # Donut chart data
    events_data = df_disaster.groupby('Disaster Type')['Events'].sum()
    deaths_data = df_disaster.groupby('Disaster Type')['Deaths'].sum()
    affected_data = df_disaster.groupby('Disaster Type')['Total Affected'].sum()

    total_events = events_data.sum()
    total_deaths = deaths_data.sum()
    total_affected = affected_data.sum()

    chart_colors = {"Floods": "blue", "Earthquake": "orange", "Epidemic": "purple"}

    def prepare_donut_data(data, colors_dict):
        non_zero_data = data[data > 0]
        total = non_zero_data.sum()
        labels = [f"{d} ({non_zero_data[d]/total*100:.1f}%)" for d in non_zero_data.index]
        colors = [colors_dict[d] for d in non_zero_data.index]
        return non_zero_data, labels, colors

    # Donut Chart 1: Events
    ax1 = fig.add_axes([0.65, 0.65, 0.2, 0.2])
    events_non_zero, events_labels, events_colors = prepare_donut_data(events_data, chart_colors)
    if len(events_non_zero) > 0:
        ax1.pie(events_non_zero, labels=events_labels, colors=events_colors,
                startangle=90, wedgeprops=dict(width=0.3), textprops=dict(fontsize=10))
    ax1.text(0, 0, f'Total Events\n{total_events}', ha='center', va='center', fontsize=12)
    ax1.set_title("Events by Disaster Type", fontsize=14)

    # Donut Chart 2: Deaths
    ax2 = fig.add_axes([0.65, 0.35, 0.2, 0.2])
    deaths_non_zero, deaths_labels, deaths_colors = prepare_donut_data(deaths_data, chart_colors)
    if len(deaths_non_zero) > 0:
        ax2.pie(deaths_non_zero, labels=deaths_labels, colors=deaths_colors,
                startangle=90, wedgeprops=dict(width=0.3), textprops=dict(fontsize=10))
    ax2.text(0, 0, f'Total Deaths\n{total_deaths}', ha='center', va='center', fontsize=12)
    ax2.set_title("Deaths by Disaster Type", fontsize=14)

    # Donut Chart 3: Total Affected
    ax3 = fig.add_axes([0.65, 0.05, 0.2, 0.2])
    affected_non_zero, affected_labels, affected_colors = prepare_donut_data(affected_data, chart_colors)
    if len(affected_non_zero) > 0:
        ax3.pie(affected_non_zero, labels=affected_labels, colors=affected_colors,
                startangle=90, wedgeprops=dict(width=0.3), textprops=dict(fontsize=10))
    ax3.text(0, 0, f'Total Affected\n{total_affected}', ha='center', va='center', fontsize=12)
    ax3.set_title("Total Affected by Disaster Type", fontsize=14)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / 'eastern_africa_disaster_map_with_donuts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: eastern_africa_disaster_map_with_donuts.png")


def main():
    """Main function to run all analyses."""
    print("=" * 60)
    print("IDMC Displacement Analysis")
    print("=" * 60)

    # Define paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    input_dir = base_dir / 'Input'
    output_dir = base_dir / 'output'
    shapefiles_dir = base_dir / 'shapefiles'
    icons_dir = base_dir / 'icons'

    shapefile_path = shapefiles_dir / 'Administrative0_Boundaries_ICPAC_Countries.shp'

    # Find and load the latest summary file
    try:
        summary_file = get_latest_summary_file(output_dir)
        df, dff = load_data(summary_file, input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run process_idmc_data.py first to generate the summary file.")
        return

    print(f"\nData Summary:")
    print(f"  Summary records: {len(df)}")
    print(f"  Raw records: {len(dff)}")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Displacement types: {df['displacement_type'].unique().tolist()}")

    # Run all visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    plot_geographic_distribution(df, shapefile_path, output_dir)
    plot_displacement_type_comparison(df, output_dir)
    plot_top_events(df, output_dir)
    plot_displacement_trend(dff, output_dir)
    plot_displacement_trend_2025(dff, output_dir)
    plot_disaster_map_with_donuts(shapefile_path, icons_dir, output_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
