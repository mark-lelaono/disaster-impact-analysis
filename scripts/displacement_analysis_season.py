"""
Season-Aware Displacement Analysis Script

Generalized from displacement_analysis_ond.py to support any season
(OND, MAM, JJAS). Can be run standalone or as part of the pipeline.

Usage:
    python scripts/displacement_analysis_season.py --season OND --year 2025
    python scripts/displacement_analysis_season.py --season MAM --year 2025
    python scripts/displacement_analysis_season.py --season JJAS --year 2025
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import glob
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    get_season_config, get_data_source_footer,
    INPUT_DIR, OUTPUT_DIR, SHAPEFILES_DIR, SHAPEFILE_PATH
)


def format_number(value, pos):
    """Format numbers with K for thousands and M for millions."""
    if value >= 1e6:
        return f'{value/1e6:.1f}M'
    elif value >= 1e3:
        return f'{value/1e3:.0f}K'
    else:
        return f'{value:.0f}'


def format_heatmap_annotation(val):
    """Format heatmap annotation with K for thousands and M for millions."""
    if val >= 1e6:
        return f'{val/1e6:.1f}M'
    elif val >= 1e3:
        return f'{val/1e3:.0f}K'
    else:
        return f'{val:.0f}'


def add_data_source_footer(fig, footer_text, y_position=-0.12):
    """Add data source footer to the bottom of a figure."""
    fig.text(0.5, y_position, footer_text,
             ha='center', va='top', fontsize=9, style='italic', color='gray')


def get_latest_summary_file(output_dir: Path) -> Path:
    """Find the most recent summary file in the output directory."""
    pattern = str(output_dir / '*_SummaryDATA_idmc_idus.xlsx')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No summary files found matching pattern: {pattern}")
    return Path(max(files, key=lambda x: Path(x).stat().st_mtime))


def load_data(summary_file: Path, input_dir: Path, season_cfg: dict):
    """Load summary and raw data files, filtered for the given season's months.

    Returns (df_season, dff_season, dff_with_pre) where dff_with_pre includes
    both pre-season and season months for trend charts.
    For ANNUAL mode, all months are included and dff_with_pre equals dff_season.
    """
    season_name = season_cfg['name']
    month_names = season_cfg['months']
    month_numbers = season_cfg['month_numbers']
    pre_month_numbers = season_cfg.get('pre_month_numbers', [])
    is_annual = season_name.upper() == 'ANNUAL'

    print(f"Loading summary data from: {summary_file}")
    df = pd.read_excel(summary_file)
    print(f"Summary data loaded: {len(df)} records")

    df_season = df[df['Month'].isin(month_names)].copy()
    label = f"Full Year" if is_annual else season_name
    print(f"{label} records: {len(df_season)}")

    raw_file = input_dir / 'idmc_idus.xlsx'
    print(f"Loading raw data from: {raw_file}")
    dff = pd.read_excel(raw_file)
    print(f"Raw data loaded: {len(dff)} records")

    dff['displacement_start_date'] = pd.to_datetime(dff['displacement_start_date'])
    dff['MonthNum'] = dff['displacement_start_date'].dt.month
    dff_season = dff[dff['MonthNum'].isin(month_numbers)].copy()
    print(f"{label} raw records: {len(dff_season)}")

    if is_annual:
        # No pre-season concept for annual — dff_with_pre is same as dff_season
        dff_with_pre = dff_season.copy()
        print(f"Annual mode: all 12 months included")
    else:
        # Pre-season + season raw data for trend charts
        all_relevant_months = sorted(set(pre_month_numbers + month_numbers))
        dff_with_pre = dff[dff['MonthNum'].isin(all_relevant_months)].copy()
        print(f"Pre-{season_name} + {season_name} raw records: {len(dff_with_pre)}")

    return df_season, dff_season, dff_with_pre


def _title_label(season_cfg):
    """Return display label for chart titles (e.g. 'OND' or 'Full Year')."""
    return "Full Year" if season_cfg['name'].upper() == 'ANNUAL' else season_cfg['name']


def plot_geographic_distribution(df, shapefile_path, output_dir, season_cfg, year, footer_text):
    """Plot geographic distribution of seasonal displacements."""
    season = season_cfg['name']
    prefix = season.lower()
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Geographic Distribution ---")

    country_figures = df.groupby('country')['figure'].sum().reset_index().sort_values(by='figure', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=country_figures, y='country', x='figure', hue='country', palette='tab10', ax=ax)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title(f"{title_label} {year} Displacements by Country", fontsize=16)
    ax.set_xlabel("Displacements", fontsize=14)
    ax.set_ylabel("Country", fontsize=14)
    add_data_source_footer(fig, footer_text)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / f'{prefix}_displacements_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_displacements_by_country.png")

    # Choropleth map
    geo_df = gpd.read_file(shapefile_path)
    country_figures_map = df.groupby("country")["figure"].sum().reset_index()
    merged = geo_df.merge(country_figures_map, how='left', left_on='COUNTRY', right_on='country')
    merged['figure'] = merged['figure'].fillna(0)
    merged['log_figure'] = np.log1p(merged['figure'])

    fig, ax = plt.subplots(figsize=(14, 9))
    merged.plot(column='log_figure', cmap='OrRd', legend=True, edgecolor='black', ax=ax)
    ax.set_title(f'{title_label} {year} Displacement Figures by Country', fontsize=16)
    ax.axis('off')
    add_data_source_footer(fig, footer_text, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_displacement_choropleth_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_displacement_choropleth_map.png")


def plot_displacement_type_comparison(df, output_dir, season_cfg, year, footer_text):
    """Plot seasonal displacement type comparison charts."""
    season = season_cfg['name']
    prefix = season.lower()
    month_order = season_cfg['months']
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Displacement Type Comparison ---")

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
    ax.set_title(f"{title_label} {year} Displacement by Type", fontsize=16)
    add_data_source_footer(fig, footer_text, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_displacement_by_type_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_displacement_by_type_pie.png")

    # Stacked Bar Chart by Month
    type_month = df.groupby(['Month', 'displacement_type'])['figure'].sum().unstack().fillna(0)
    available_months = [m for m in month_order if m in type_month.index]
    type_month = type_month.reindex(available_months)

    fig, ax = plt.subplots(figsize=(12, 7))
    type_month.plot(kind='bar', stacked=True, ax=ax)
    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title(f"{title_label} {year} Monthly Displacement by Type", fontsize=16)
    ax.set_ylabel("Displacements", fontsize=14)
    ax.set_xlabel("Month", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(title="Displacement Type", fontsize=12, title_fontsize=13)
    add_data_source_footer(fig, footer_text)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_dir / f'{prefix}_monthly_displacement_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_monthly_displacement_by_type.png")


def plot_disaster_type_breakdown(df, output_dir, season_cfg, year, footer_text):
    """Plot breakdown by disaster type for the season."""
    season = season_cfg['name']
    prefix = season.lower()
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Disaster Type Breakdown ---")

    if 'Disaster_Type' not in df.columns:
        print("Warning: Disaster_Type column not found. Skipping disaster type breakdown.")
        return

    disaster_figures = df.groupby('Disaster_Type')['figure'].sum().reset_index().sort_values(by='figure', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=disaster_figures, x='Disaster_Type', y='figure', hue='Disaster_Type', palette='Set2', ax=ax)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title(f"{title_label} {year} Displacements by Disaster Type", fontsize=16)
    ax.set_xlabel("Disaster Type", fontsize=14)
    ax.set_ylabel("Displacements", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    add_data_source_footer(fig, footer_text)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(output_dir / f'{prefix}_displacements_by_disaster_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_displacements_by_disaster_type.png")

    # Pie chart of disaster types with legend
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(disaster_figures)))
    total = disaster_figures['figure'].sum()
    percentages = (disaster_figures['figure'] / total * 100).values

    wedges, texts = ax.pie(
        disaster_figures['figure'],
        startangle=140,
        colors=colors,
        radius=0.6
    )

    legend_labels = [f"{dtype} ({pct:.1f}%)"
                     for dtype, pct in zip(disaster_figures['Disaster_Type'], percentages)]
    ax.legend(wedges, legend_labels, title="Disaster Types", loc='lower right',
              bbox_to_anchor=(1.15, 0.0), fontsize=10, title_fontsize=11)

    ax.set_title(f"{title_label} {year} Displacement by Disaster Type", fontsize=16)
    add_data_source_footer(fig, footer_text, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_disaster_type_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_disaster_type_pie.png")


def plot_top_events(df, output_dir, season_cfg, year, footer_text):
    """Plot top 10 events causing displacement for the season."""
    season = season_cfg['name']
    prefix = season.lower()
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Event-Level Insights ---")

    top_events = df.groupby('event_name')['figure'].sum().sort_values(ascending=False).head(10).reset_index()

    fig, ax = plt.subplots(figsize=(20, 11))
    sns.barplot(data=top_events, y='event_name', hue='event_name', x='figure', palette='tab10', ax=ax)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title(f"Top 10 {title_label} {year} Events Causing Displacement", fontsize=30)
    ax.set_xlabel("Displacements", fontsize=30)
    ax.set_ylabel("Event Name", fontsize=30)
    ax.legend([], [], frameon=False)
    add_data_source_footer(fig, footer_text, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_top_10_events.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_top_10_events.png")


def plot_trend_over_years(dff, output_dir, season_cfg, footer_text):
    """Plot seasonal displacement trends across multiple years."""
    season = season_cfg['name']
    prefix = season.lower()
    month_order = season_cfg['months']
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Trends Over Years ---")

    dff['Year'] = dff['displacement_start_date'].dt.year
    dff['MonthName'] = dff['displacement_start_date'].dt.strftime('%B')

    yearly = dff.groupby(['Year', 'MonthName'])['figure'].sum().reset_index()
    pivot = yearly.pivot(index='MonthName', columns='Year', values='figure').fillna(0)

    available_months = [m for m in month_order if m in pivot.index]
    pivot = pivot.reindex(available_months)

    fig, ax = plt.subplots(figsize=(12, 7))
    for year in pivot.columns:
        ax.plot(pivot.index, pivot[year], marker='o', linewidth=2, markersize=8, label=str(year))

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title(f"{title_label} Displacement Trends Across Years", fontsize=16)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Displacements (log scale)", fontsize=14)
    ax.legend(title='Year', fontsize=12, title_fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    add_data_source_footer(fig, footer_text)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / f'{prefix}_trend_across_years.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_trend_across_years.png")

    # Bar chart comparing season totals by year
    totals = dff.groupby('Year')['figure'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(totals['Year'].astype(str), totals['figure'], color='coral', edgecolor='black')

    for bar, val in zip(bars, totals['figure']):
        height = bar.get_height()
        label = f'{val/1e6:.1f}M' if val >= 1e6 else f'{val/1e3:.0f}K' if val >= 1e3 else f'{val:.0f}'
        ax.text(bar.get_x() + bar.get_width()/2., height, label,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_title(f"Total {title_label} Displacements by Year", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Total Displacements", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    add_data_source_footer(fig, footer_text)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / f'{prefix}_total_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_total_by_year.png")


def plot_pre_season_trend(dff, output_dir, season_cfg, year, footer_text):
    """Plot displacement trend for Pre-season + season months of the given year.

    For ANNUAL mode, plots all 12 months as a single continuous line (no pre-season split).
    """
    season = season_cfg['name']
    prefix = season.lower()
    is_annual = season.upper() == 'ANNUAL'
    season_months = season_cfg['month_numbers']
    pre_months = season_cfg.get('pre_month_numbers', [])
    allowed_months = sorted(set(pre_months + season_months))

    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Monthly Trend ({year}) ---")

    dff['Year'] = dff['displacement_start_date'].dt.year
    dff['MonthNum'] = dff['displacement_start_date'].dt.month
    dff['MonthName'] = dff['displacement_start_date'].dt.strftime('%B')

    # Filter to specified year and allowed months only
    dff_year = dff[(dff['Year'] == year) & (dff['MonthNum'].isin(allowed_months))].copy()

    monthly = dff_year.groupby(['MonthNum', 'MonthName'])['figure'].sum().reset_index()
    monthly = monthly.sort_values('MonthNum')

    if monthly.empty:
        print(f"No data for {title_label} {year}. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    if is_annual:
        # Single continuous line for full year
        ax.plot(monthly['MonthName'], monthly['figure'], marker='o', color='coral',
                label=f'{year}', linewidth=2, markersize=8)
        ax.set_title(f"Monthly Displacement Trend - {year}", fontsize=16)
        chart_filename = f'{prefix}_monthly_trend_{year}.png'
    else:
        first_season_month = min(season_months)

        # Pre-season (blue)
        data_pre = monthly[monthly['MonthNum'] < first_season_month]
        if not data_pre.empty:
            ax.plot(data_pre['MonthName'], data_pre['figure'], marker='o', color='blue',
                    label=f'Pre-{season} {year}', linewidth=2, markersize=8)

        # Season (red)
        data_season = monthly[monthly['MonthNum'] >= first_season_month]
        if not data_season.empty:
            ax.plot(data_season['MonthName'], data_season['figure'], marker='o', color='red',
                    label=f'{season} {year}', linewidth=2, markersize=8)

        # Connect last pre to first season point
        if not data_pre.empty and not data_season.empty:
            ax.plot(
                [data_pre.iloc[-1]['MonthName'], data_season.iloc[0]['MonthName']],
                [data_pre.iloc[-1]['figure'], data_season.iloc[0]['figure']],
                color='blue', linestyle='-', linewidth=2
            )

        ax.set_title(f"Pre-{season} & {season} Displacement Trend - {year}", fontsize=16)
        chart_filename = f'{prefix}_pre_season_trend_{year}.png'

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax.set_xlabel(f"{year} Months", fontsize=14)
    ax.set_ylabel("Number of Displacements", fontsize=14)
    ax.legend(fontsize=12, title='Period', title_fontsize=13)
    plt.xticks(rotation=45, fontsize=12)
    add_data_source_footer(fig, footer_text)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_dir / chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart_filename}")


def plot_country_month_heatmap(df, output_dir, season_cfg, year, footer_text):
    """Plot heatmap of displacements by country and season month."""
    season = season_cfg['name']
    prefix = season.lower()
    month_order = season_cfg['months']
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Country-Month Heatmap ---")

    # Abbreviated month names for x-axis labels
    month_abbrev = {
        'January': 'Jan', 'February': 'Feb', 'March': 'Mar',
        'April': 'Apr', 'May': 'May', 'June': 'June',
        'July': 'July', 'August': 'Aug', 'September': 'Sept',
        'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
    }

    heatmap_data = df.pivot_table(values='figure', index='country', columns='Month', aggfunc='sum', fill_value=0)
    available_months = [m for m in month_order if m in heatmap_data.columns]
    heatmap_data = heatmap_data[available_months]
    heatmap_data.columns = [month_abbrev.get(m, m) for m in heatmap_data.columns]

    heatmap_data['Total'] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values('Total', ascending=False)
    heatmap_data = heatmap_data.drop('Total', axis=1)

    annot_data = heatmap_data.map(format_heatmap_annotation)

    n_countries = len(heatmap_data)
    fig_height = max(8, n_countries * 0.9 + 3)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    sns.heatmap(heatmap_data, annot=annot_data, fmt='', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Displacements'}, ax=ax,
                annot_kws={'fontsize': 18, 'fontweight': 'bold'})
    ax.set_title(f"{title_label} {year} Displacements: Country vs Month", fontsize=22, fontweight='bold')
    ax.set_xlabel("Month", fontsize=18)
    ax.set_ylabel("Country", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=14, labelrotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_number))
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Displacements', fontsize=16)
    add_data_source_footer(fig, footer_text, y_position=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_country_month_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_country_month_heatmap.png")


def generate_summary_stats(df, dff, output_dir, season_cfg, year):
    """Generate and save seasonal summary statistics."""
    season = season_cfg['name']
    prefix = season.lower()
    month_order = season_cfg['months']
    title_label = _title_label(season_cfg)
    print(f"\n--- {title_label} Summary Statistics ---")

    total_displaced = df['figure'].sum()
    num_events = df['event_name'].nunique()
    num_countries = df['country'].nunique()
    top_country = df.groupby('country')['figure'].sum().idxmax()
    top_country_figure = df.groupby('country')['figure'].sum().max()

    type_breakdown = df.groupby('displacement_type')['figure'].sum()
    monthly_breakdown = df.groupby('Month')['figure'].sum()

    summary = f"""
{title_label} {year} DISPLACEMENT ANALYSIS SUMMARY
{'='*50}

Total Displaced ({title_label}): {total_displaced:,.0f}
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
    for month in month_order:
        if month in monthly_breakdown.index:
            figure = monthly_breakdown[month]
            pct = (figure / total_displaced) * 100
            summary += f"  {month}: {figure:,.0f} ({pct:.1f}%)\n"

    print(summary)

    with open(output_dir / f'{prefix}_summary_statistics.txt', 'w') as f:
        f.write(summary)
    print(f"Saved: {prefix}_summary_statistics.txt")


def run_seasonal_analysis(
    season: str,
    year: int,
    summary_file: Path = None,
    input_dir: Path = None,
    output_dir: Path = None,
    shapefile_path: Path = None,
) -> pd.DataFrame:
    """
    Run the full seasonal analysis pipeline.

    Returns the season-filtered DataFrame for DB storage.
    """
    season_cfg = get_season_config(season)
    input_dir = input_dir or INPUT_DIR
    output_dir = output_dir or OUTPUT_DIR
    shapefile_path = shapefile_path or SHAPEFILE_PATH
    footer_text = get_data_source_footer(season, year)

    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_file is None:
        summary_file = get_latest_summary_file(output_dir)

    is_annual = season.upper() == 'ANNUAL'
    title_label = "Full Year" if is_annual else season

    print("=" * 60)
    print(f"IDMC {title_label} ({season_cfg['long_name']}) Displacement Analysis")
    print(f"Year: {year}")
    print("=" * 60)

    df_season, dff_season, dff_with_pre = load_data(summary_file, input_dir, season_cfg)

    if len(df_season) == 0:
        print(f"Warning: No {title_label} data found in the summary file.")
        return df_season

    print(f"\n{title_label} Data Summary:")
    print(f"  Summary records: {len(df_season)}")
    print(f"  Raw records: {len(dff_season)}")
    if not is_annual:
        print(f"  Pre-{season} + {season} raw records: {len(dff_with_pre)}")
    print(f"  Countries: {df_season['country'].nunique()}")
    print(f"  Displacement types: {df_season['displacement_type'].unique().tolist()}")
    print(f"  Months: {df_season['Month'].unique().tolist()}")

    print(f"\nGenerating {title_label} Visualizations")
    print("=" * 60)

    plot_geographic_distribution(df_season, shapefile_path, output_dir, season_cfg, year, footer_text)
    plot_displacement_type_comparison(df_season, output_dir, season_cfg, year, footer_text)
    plot_disaster_type_breakdown(df_season, output_dir, season_cfg, year, footer_text)
    plot_top_events(df_season, output_dir, season_cfg, year, footer_text)
    plot_trend_over_years(dff_season, output_dir, season_cfg, footer_text)
    plot_pre_season_trend(dff_with_pre, output_dir, season_cfg, year, footer_text)
    plot_country_month_heatmap(df_season, output_dir, season_cfg, year, footer_text)
    generate_summary_stats(df_season, dff_season, output_dir, season_cfg, year)

    print(f"\n{title_label} Analysis Complete!")
    print(f"All visualizations saved to: {output_dir}")

    return df_season


def main():
    parser = argparse.ArgumentParser(description="Seasonal Displacement Analysis")
    parser.add_argument('--season', required=True, choices=['OND', 'MAM', 'JJAS', 'ANNUAL'],
                        help='Season to analyze (use ANNUAL for full-year analysis)')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze')
    args = parser.parse_args()

    run_seasonal_analysis(args.season, args.year)


if __name__ == '__main__':
    main()
