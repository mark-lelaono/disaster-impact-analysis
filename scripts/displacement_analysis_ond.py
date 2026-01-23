"""
OND (October-November-December) Displacement Analysis Script

This script performs analysis of IDMC displacement data specifically for the
OND season (October, November, December). It reads the summary data from the
output folder and saves all visualizations there as well.

Usage:
    python displacement_analysis_ond.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
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


# Data source footer text
DATA_SOURCE_FOOTER = "Data sources: ECHO, FEWS NET, IOM, UNOCHA, IPC, IFRC, WHO | Analysis period: October–December 2025"


def add_data_source_footer(fig, y_position=-0.12):
    """Add data source footer to the bottom of a figure."""
    fig.text(0.5, y_position, DATA_SOURCE_FOOTER,
             ha='center', va='top', fontsize=9, style='italic', color='gray')


def get_latest_summary_file(output_dir: Path) -> Path:
    """Find the most recent summary file in the output directory."""
    pattern = str(output_dir / '*_SummaryDATA_idmc_idus.xlsx')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No summary files found matching pattern: {pattern}")
    # Return the most recently modified file
    return Path(max(files, key=lambda x: Path(x).stat().st_mtime))


def load_data(summary_file: Path, input_dir: Path):
    """Load summary and raw data files, filtered for OND months."""
    print(f"Loading summary data from: {summary_file}")
    df = pd.read_excel(summary_file)
    print(f"Summary data loaded: {len(df)} records")

    # Filter for OND months only
    ond_months = ['October', 'November', 'December']
    df_ond = df[df['Month'].isin(ond_months)].copy()
    print(f"OND records (Oct-Nov-Dec): {len(df_ond)} records")

    # Load raw data for trend analysis
    raw_file = input_dir / 'idmc_idus.xlsx'
    print(f"Loading raw data from: {raw_file}")
    dff = pd.read_excel(raw_file)
    print(f"Raw data loaded: {len(dff)} records")

    # Filter raw data for OND months
    dff['displacement_start_date'] = pd.to_datetime(dff['displacement_start_date'])
    dff['MonthNum'] = dff['displacement_start_date'].dt.month
    dff_ond = dff[dff['MonthNum'].isin([10, 11, 12])].copy()
    print(f"OND raw records: {len(dff_ond)} records")

    return df_ond, dff_ond


def plot_geographic_distribution(df: pd.DataFrame, shapefile_path: Path, output_dir: Path):
    """Plot geographic distribution of OND displacements."""
    print("\n--- OND Geographic Distribution ---")

    # Displacements by country (log-scaled bar plot)
    country_figures = df.groupby('country')['figure'].sum().reset_index().sort_values(by='figure', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=country_figures, y='country', x='figure', hue='country', palette='tab10', ax=ax)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title("OND Displacements by Country", fontsize=16)
    ax.set_xlabel("Displacements", fontsize=14)
    ax.set_ylabel("Country", fontsize=14)
    add_data_source_footer(fig)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / 'ond_displacements_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_displacements_by_country.png")

    # Choropleth map
    geo_df = gpd.read_file(shapefile_path)
    country_figures_map = df.groupby("country")["figure"].sum().reset_index()
    merged = geo_df.merge(country_figures_map, how='left', left_on='COUNTRY', right_on='country')
    merged['figure'] = merged['figure'].fillna(0)
    merged['log_figure'] = np.log1p(merged['figure'])

    fig, ax = plt.subplots(figsize=(14, 9))
    merged.plot(column='log_figure', cmap='OrRd', legend=True, edgecolor='black', ax=ax)
    ax.set_title('OND Displacement Figures by Country', fontsize=16)
    ax.axis('off')
    add_data_source_footer(fig, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ond_displacement_choropleth_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_displacement_choropleth_map.png")


def plot_displacement_type_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot OND displacement type comparison charts."""
    print("\n--- OND Displacement Type Comparison ---")

    # Pie Chart
    type_summary = df.groupby('displacement_type')['figure'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(
        type_summary['figure'],
        labels=type_summary['displacement_type'],
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 12}
    )
    ax.set_title("OND Displacement by Type", fontsize=16)
    add_data_source_footer(fig, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ond_displacement_by_type_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_displacement_by_type_pie.png")

    # Stacked Bar Chart by Month (Oct, Nov, Dec)
    month_order = ['October', 'November', 'December']

    type_month = df.groupby(['Month', 'displacement_type'])['figure'].sum().unstack().fillna(0)

    # Reindex to order months (Oct, Nov, Dec)
    available_months = [m for m in month_order if m in type_month.index]
    type_month = type_month.reindex(available_months)

    fig, ax = plt.subplots(figsize=(12, 7))
    type_month.plot(kind='bar', stacked=True, ax=ax)

    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title("OND Monthly Displacement by Type", fontsize=16)
    ax.set_ylabel("Displacements", fontsize=14)
    ax.set_xlabel("Month", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(title="Displacement Type", fontsize=12, title_fontsize=13)
    add_data_source_footer(fig)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_dir / 'ond_monthly_displacement_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_monthly_displacement_by_type.png")


def plot_disaster_type_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot breakdown by disaster type for OND season."""
    print("\n--- OND Disaster Type Breakdown ---")

    # Check if Disaster_Type column exists
    if 'Disaster_Type' not in df.columns:
        print("Warning: Disaster_Type column not found. Skipping disaster type breakdown.")
        return

    # Bar chart of disaster types
    disaster_figures = df.groupby('Disaster_Type')['figure'].sum().reset_index().sort_values(by='figure', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=disaster_figures, x='Disaster_Type', y='figure', hue='Disaster_Type', palette='Set2', ax=ax)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title("OND Displacements by Disaster Type", fontsize=16)
    ax.set_xlabel("Disaster Type", fontsize=14)
    ax.set_ylabel("Displacements", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    add_data_source_footer(fig)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(output_dir / 'ond_displacements_by_disaster_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_displacements_by_disaster_type.png")

    # Pie chart of disaster types with legend
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(disaster_figures)))

    # Calculate percentages for legend
    total = disaster_figures['figure'].sum()
    percentages = (disaster_figures['figure'] / total * 100).values

    # Create pie chart without labels (labels will be in legend)
    wedges, texts = ax.pie(
        disaster_figures['figure'],
        startangle=140,
        colors=colors,
        radius=0.6  # Smaller pie chart size to create more space for legend
    )

    # Create legend labels with disaster type and percentage
    legend_labels = [f"{dtype} ({pct:.1f}%)"
                     for dtype, pct in zip(disaster_figures['Disaster_Type'], percentages)]

    # Add legend in bottom right with offset to avoid touching pie chart
    ax.legend(wedges, legend_labels, title="Disaster Types", loc='lower right',
              bbox_to_anchor=(1.15, 0.0), fontsize=10, title_fontsize=11)

    ax.set_title("OND Displacement by Disaster Type", fontsize=16)
    add_data_source_footer(fig, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ond_disaster_type_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_disaster_type_pie.png")


def plot_top_events(df: pd.DataFrame, output_dir: Path):
    """Plot top 10 OND events causing displacement."""
    print("\n--- OND Event-Level Insights ---")

    top_events = df.groupby('event_name')['figure'].sum().sort_values(ascending=False).head(10).reset_index()

    fig, ax = plt.subplots(figsize=(20, 11))
    sns.barplot(data=top_events, y='event_name', hue='event_name', x='figure', palette='tab10', ax=ax)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title("Top 10 OND Events Causing Displacement", fontsize=30)
    ax.set_xlabel("Displacements", fontsize=30)
    ax.set_ylabel("Event Name", fontsize=30)
    ax.legend([], [], frameon=False)
    add_data_source_footer(fig, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ond_top_10_events.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_top_10_events.png")


def plot_ond_trend_over_years(dff: pd.DataFrame, output_dir: Path):
    """Plot OND displacement trends across multiple years."""
    print("\n--- OND Trends Over Years ---")

    # Add year column
    dff['Year'] = dff['displacement_start_date'].dt.year
    dff['MonthName'] = dff['displacement_start_date'].dt.strftime('%B')

    # Group by Year and Month
    ond_yearly = dff.groupby(['Year', 'MonthName'])['figure'].sum().reset_index()

    # Pivot for line plot
    ond_pivot = ond_yearly.pivot(index='MonthName', columns='Year', values='figure').fillna(0)

    # Reorder months (Oct, Nov, Dec)
    month_order = ['October', 'November', 'December']
    available_months = [m for m in month_order if m in ond_pivot.index]
    ond_pivot = ond_pivot.reindex(available_months)

    fig, ax = plt.subplots(figsize=(12, 7))
    for year in ond_pivot.columns:
        ax.plot(ond_pivot.index, ond_pivot[year], marker='o', linewidth=2, markersize=8, label=str(year))

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title("OND Displacement Trends Across Years", fontsize=16)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Displacements (log scale)", fontsize=14)
    ax.legend(title='Year', fontsize=12, title_fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    add_data_source_footer(fig)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / 'ond_trend_across_years.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_trend_across_years.png")

    # Bar chart comparing OND totals by year
    ond_totals = dff.groupby('Year')['figure'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(ond_totals['Year'].astype(str), ond_totals['figure'], color='coral', edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, ond_totals['figure']):
        height = bar.get_height()
        label = f'{val/1e6:.1f}M' if val >= 1e6 else f'{val/1e3:.0f}K' if val >= 1e3 else f'{val:.0f}'
        ax.text(bar.get_x() + bar.get_width()/2., height, label,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title("Total OND Displacements by Year", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Total Displacements", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    add_data_source_footer(fig)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / 'ond_total_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_total_by_year.png")


def format_heatmap_annotation(val):
    """Format heatmap annotation with K for thousands and M for millions."""
    if val >= 1e6:
        return f'{val/1e6:.1f}M'
    elif val >= 1e3:
        return f'{val/1e3:.0f}K'
    else:
        return f'{val:.0f}'


def plot_country_month_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of displacements by country and OND month."""
    print("\n--- OND Country-Month Heatmap ---")

    # Pivot table for heatmap
    month_order = ['October', 'November', 'December']
    heatmap_data = df.pivot_table(values='figure', index='country', columns='Month', aggfunc='sum', fill_value=0)

    # Reorder columns
    available_months = [m for m in month_order if m in heatmap_data.columns]
    heatmap_data = heatmap_data[available_months]

    # Sort by total
    heatmap_data['Total'] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values('Total', ascending=False)
    heatmap_data = heatmap_data.drop('Total', axis=1)

    # Create formatted annotations for K/M display
    annot_data = heatmap_data.applymap(format_heatmap_annotation)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=annot_data, fmt='', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Displacements'}, ax=ax,
                annot_kws={'fontsize': 18, 'fontweight': 'bold'})
    ax.set_title("OND Displacements: Country vs Month", fontsize=22, fontweight='bold')
    ax.set_xlabel("Month", fontsize=18)
    ax.set_ylabel("Country", fontsize=18)
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # Increase colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Displacements', fontsize=16)
    add_data_source_footer(fig, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ond_country_month_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ond_country_month_heatmap.png")


def generate_ond_summary_stats(df: pd.DataFrame, dff: pd.DataFrame, output_dir: Path):
    """Generate and save OND summary statistics."""
    print("\n--- OND Summary Statistics ---")

    # Calculate summary statistics
    total_displaced = df['figure'].sum()
    num_events = df['event_name'].nunique()
    num_countries = df['country'].nunique()
    top_country = df.groupby('country')['figure'].sum().idxmax()
    top_country_figure = df.groupby('country')['figure'].sum().max()

    # Displacement type breakdown
    type_breakdown = df.groupby('displacement_type')['figure'].sum()

    # Monthly breakdown
    monthly_breakdown = df.groupby('Month')['figure'].sum()

    # Create summary text
    summary = f"""
OND DISPLACEMENT ANALYSIS SUMMARY
{'='*50}

Total Displaced (OND): {total_displaced:,.0f}
Number of Events: {num_events}
Countries Affected: {num_countries}

Top Affected Country: {top_country} ({top_country_figure:,.0f} displaced)

DISPLACEMENT BY TYPE:
{'-'*30}
"""
    for dtype, figure in type_breakdown.items():
        pct = (figure / total_displaced) * 100
        summary += f"  {dtype}: {figure:,.0f} ({pct:.1f}%)\n"

    summary += f"""
MONTHLY BREAKDOWN:
{'-'*30}
"""
    for month in ['October', 'November', 'December']:
        if month in monthly_breakdown.index:
            figure = monthly_breakdown[month]
            pct = (figure / total_displaced) * 100
            summary += f"  {month}: {figure:,.0f} ({pct:.1f}%)\n"

    print(summary)

    # Save to file
    with open(output_dir / 'ond_summary_statistics.txt', 'w') as f:
        f.write(summary)
    print(f"Saved: ond_summary_statistics.txt")


def main():
    """Main function to run OND analysis."""
    print("=" * 60)
    print("IDMC OND (Oct-Nov-Dec) Displacement Analysis")
    print("=" * 60)

    # Define paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    input_dir = base_dir / 'Input'
    output_dir = base_dir / 'output'
    shapefiles_dir = base_dir / 'shapefiles'

    shapefile_path = shapefiles_dir / 'Administrative0_Boundaries_ICPAC_Countries.shp'

    # Find and load the latest summary file (filtered for OND)
    try:
        summary_file = get_latest_summary_file(output_dir)
        df_ond, dff_ond = load_data(summary_file, input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run process_idmc_data.py first to generate the summary file.")
        return

    if len(df_ond) == 0:
        print("Warning: No OND data found in the summary file.")
        print("The summary file may not contain October, November, or December records.")
        return

    print(f"\nOND Data Summary:")
    print(f"  Summary records (OND): {len(df_ond)}")
    print(f"  Raw records (OND): {len(dff_ond)}")
    print(f"  Countries: {df_ond['country'].nunique()}")
    print(f"  Displacement types: {df_ond['displacement_type'].unique().tolist()}")
    print(f"  Months: {df_ond['Month'].unique().tolist()}")

    # Run all visualizations
    print("\n" + "=" * 60)
    print("Generating OND Visualizations")
    print("=" * 60)

    plot_geographic_distribution(df_ond, shapefile_path, output_dir)
    plot_displacement_type_comparison(df_ond, output_dir)
    plot_disaster_type_breakdown(df_ond, output_dir)
    plot_top_events(df_ond, output_dir)
    plot_ond_trend_over_years(dff_ond, output_dir)
    plot_country_month_heatmap(df_ond, output_dir)
    generate_ond_summary_stats(df_ond, dff_ond, output_dir)

    print("\n" + "=" * 60)
    print("OND Analysis Complete!")
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
