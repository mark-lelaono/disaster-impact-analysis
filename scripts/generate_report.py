"""
HTML Report Generator for Disaster Displacement Analysis.

Generates a self-contained HTML report with embedded analysis images
for any season (ANNUAL, MAM, JJAS, OND) and year.

Usage:
    # Standalone
    python scripts/generate_report.py --season ANNUAL --year 2025
    python scripts/generate_report.py --season OND --year 2025

    # As part of the pipeline (called by run_analysis.py)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from html import escape

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    get_season_config, OUTPUT_DIR, PROJECT_ROOT,
    DATA_SOURCE_ORGANIZATIONS,
)


# ============================================================
# DATA LOADING
# ============================================================

def _load_summary_stats(output_dir: Path, prefix: str) -> dict:
    """Parse the summary statistics text file into a dict."""
    stats_file = output_dir / f'{prefix}_summary_statistics.txt'
    if not stats_file.exists():
        return {}

    text = stats_file.read_text()
    stats = {'raw': text}

    for line in text.splitlines():
        line = line.strip()
        if line.startswith('Total Displaced'):
            stats['total_displaced'] = line.split(':')[1].strip()
        elif line.startswith('Number of Events'):
            stats['num_events'] = line.split(':')[1].strip()
        elif line.startswith('Countries Affected'):
            stats['num_countries'] = line.split(':')[1].strip()
        elif line.startswith('Top Affected Country'):
            stats['top_country'] = line.split(':')[1].strip()

    # Parse displacement by type
    stats['type_breakdown'] = {}
    stats['monthly_breakdown'] = {}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if 'DISPLACEMENT BY TYPE' in line:
            section = 'type'
            continue
        elif 'MONTHLY BREAKDOWN' in line:
            section = 'monthly'
            continue
        elif line.startswith('---'):
            continue

        if section == 'type' and ':' in line and line.startswith(('Conflict', 'Disaster')):
            parts = line.split(':')
            name = parts[0].strip()
            val = parts[1].strip()
            stats['type_breakdown'][name] = val
        elif section == 'monthly' and ':' in line:
            parts = line.split(':')
            name = parts[0].strip()
            val = ':'.join(parts[1:]).strip()
            stats['monthly_breakdown'][name] = val

    return stats


def _load_summary_excel(output_dir: Path, prefix: str) -> pd.DataFrame:
    """Load the summary Excel file for detailed breakdowns."""
    excel_file = output_dir / f'{prefix.upper()}_SummaryDATA_idmc_idus.xlsx'
    if not excel_file.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(excel_file)
    except Exception:
        return pd.DataFrame()


def _compute_country_table(df: pd.DataFrame) -> list[dict]:
    """Compute country-level statistics from summary dataframe."""
    if df.empty or 'country' not in df.columns or 'figure' not in df.columns:
        return []

    country_totals = df.groupby('country')['figure'].sum().sort_values(ascending=False)
    grand_total = country_totals.sum()
    rows = []
    for rank, (country, total) in enumerate(country_totals.items(), 1):
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        # Determine primary driver
        if 'displacement_type' in df.columns:
            ct = df[df['country'] == country].groupby('displacement_type')['figure'].sum()
            if len(ct) == 1:
                driver = f"{ct.index[0]} (100%)"
            elif len(ct) > 1:
                top = ct.idxmax()
                top_pct = ct.max() / ct.sum() * 100
                driver = f"{top} ({top_pct:.0f}%)"
            else:
                driver = "N/A"
        else:
            driver = "N/A"

        events = df[df['country'] == country]['event_name'].nunique() if 'event_name' in df.columns else 0
        rows.append({
            'rank': rank, 'country': country, 'total': int(total),
            'pct': pct, 'events': events, 'driver': driver,
        })
    return rows


def _compute_disaster_table(df: pd.DataFrame) -> list[dict]:
    """Compute disaster type breakdown from summary dataframe."""
    if df.empty:
        return []

    disaster_df = df[df['displacement_type'] == 'Disaster'] if 'displacement_type' in df.columns else df
    if disaster_df.empty or 'Disaster_Type' not in disaster_df.columns:
        return []

    dtype_totals = disaster_df.groupby('Disaster_Type')['figure'].sum().sort_values(ascending=False)
    dtype_totals = dtype_totals[dtype_totals.index.notna() & (dtype_totals.index != '')]
    grand_total = dtype_totals.sum()
    rows = []
    for dtype, total in dtype_totals.items():
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        events = disaster_df[disaster_df['Disaster_Type'] == dtype]['event_name'].nunique() if 'event_name' in disaster_df.columns else 0
        rows.append({
            'type': dtype, 'total': int(total),
            'pct': pct, 'events': events,
        })
    return rows


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def _get_season_images(output_dir: Path, prefix: str, year: int) -> dict:
    """Discover available images for a season prefix.

    Returns dict of key -> relative path (from reports/ to output/).
    """
    # For ANNUAL the monthly trend file uses a different name pattern
    if prefix == 'annual':
        trend_file = f'{prefix}_monthly_trend_{year}.png'
    else:
        trend_file = f'{prefix}_pre_season_trend_{year}.png'

    image_keys = {
        'displacements_by_country': f'{prefix}_displacements_by_country.png',
        'displacement_by_type_pie': f'{prefix}_displacement_by_type_pie.png',
        'displacements_by_disaster_type': f'{prefix}_displacements_by_disaster_type.png',
        'disaster_map_with_donuts': f'{prefix}_disaster_map_with_donuts.png',
        'monthly_trend': trend_file,
        'country_month_heatmap': f'{prefix}_country_month_heatmap.png',
        'displacement_choropleth_map': f'{prefix}_displacement_choropleth_map.png',
        'disaster_type_pie': f'{prefix}_disaster_type_pie.png',
        'top_10_events': f'{prefix}_top_10_events.png',
        'total_by_year': f'{prefix}_total_by_year.png',
        'trend_across_years': f'{prefix}_trend_across_years.png',
        'monthly_displacement_by_type': f'{prefix}_monthly_displacement_by_type.png',
    }
    # Relative path from reports/ directory up to output/
    rel_prefix = '../output/'
    found = {}
    for key, filename in image_keys.items():
        if (output_dir / filename).exists():
            found[key] = rel_prefix + filename
    return found


# ============================================================
# HTML GENERATION
# ============================================================

_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700;900&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
:root {
    --primary: #1a3a5c; --secondary: #2c6fbb; --accent: #e67e22;
    --danger: #c0392b; --success: #27ae60; --bg-light: #f8f9fa;
    --bg-section: #eef3f8; --text: #2c3e50; --text-light: #5d6d7e; --border: #d5dce4;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Source Sans Pro', 'Segoe UI', Tahoma, sans-serif; color: var(--text); line-height: 1.7; background: #fff; }
.cover { background: linear-gradient(135deg, var(--primary) 0%, #0d2137 100%); color: white; min-height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 60px 40px; page-break-after: always; }
.cover-content { max-width: 800px; }
.cover-badge { display: inline-block; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3); border-radius: 30px; padding: 8px 24px; font-size: 14px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px; }
.cover h1 { font-family: 'Merriweather', serif; font-size: 42px; font-weight: 900; line-height: 1.2; margin-bottom: 20px; }
.cover h2 { font-size: 22px; font-weight: 300; opacity: 0.9; margin-bottom: 40px; }
.cover-stats { display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; margin-top: 50px; }
.cover-stat { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 12px; padding: 20px 30px; min-width: 160px; }
.cover-stat .number { font-size: 32px; font-weight: 700; color: #f39c12; }
.cover-stat .desc { font-size: 13px; opacity: 0.8; margin-top: 4px; }
.cover-meta { display: flex; gap: 40px; justify-content: center; flex-wrap: wrap; margin-top: 40px; padding-top: 40px; border-top: 1px solid rgba(255,255,255,0.2); }
.cover-meta-item { text-align: center; }
.cover-meta-item .label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; }
.cover-meta-item .value { font-size: 18px; font-weight: 600; margin-top: 4px; }
.container { max-width: 1100px; margin: 0 auto; padding: 40px 30px; }
section { margin-bottom: 50px; }
h2.section-title { font-family: 'Merriweather', serif; font-size: 28px; color: var(--primary); margin-bottom: 25px; padding-bottom: 12px; border-bottom: 3px solid var(--secondary); page-break-after: avoid; }
h3.sub-title { font-family: 'Merriweather', serif; font-size: 20px; color: var(--primary); margin: 30px 0 15px 0; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
p { margin-bottom: 15px; font-size: 15.5px; text-align: justify; }
.key-figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 25px 0; }
.key-figure { background: linear-gradient(135deg, var(--bg-section), #fff); border-left: 4px solid var(--secondary); border-radius: 8px; padding: 20px; text-align: center; }
.key-figure.danger { border-left-color: var(--danger); }
.key-figure.warning { border-left-color: var(--accent); }
.key-figure.success { border-left-color: var(--success); }
.key-figure .number { font-size: 30px; font-weight: 700; color: var(--primary); }
.key-figure .label { font-size: 13px; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 5px; }
.callout { background: var(--bg-section); border-left: 4px solid var(--secondary); border-radius: 0 8px 8px 0; padding: 20px 25px; margin: 20px 0; }
.callout.alert { background: #fdf2f2; border-left-color: var(--danger); }
.callout.highlight { background: #fef9e7; border-left-color: var(--accent); }
.callout strong { color: var(--primary); }
.figure-container { margin: 30px 0; text-align: center; page-break-inside: avoid; }
.figure-container img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
.figure-caption { font-size: 13px; color: var(--text-light); font-style: italic; margin-top: 12px; padding: 0 40px; }
.figure-label { display: inline-block; background: var(--primary); color: white; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 4px; margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }
thead th { background: var(--primary); color: white; padding: 12px 15px; text-align: left; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
tbody td { padding: 10px 15px; border-bottom: 1px solid var(--border); }
tbody tr:nth-child(even) { background: var(--bg-light); }
tbody tr:hover { background: #e8f0fe; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.hl { background: #fef9e7 !important; font-weight: 600; }
.season-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin: 25px 0; }
.season-card { background: white; border: 1px solid var(--border); border-radius: 10px; padding: 25px 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.season-card h4 { color: var(--primary); font-size: 16px; margin-bottom: 15px; }
.season-card .stat { font-size: 26px; font-weight: 700; color: var(--secondary); }
.season-card .stat-label { font-size: 12px; color: var(--text-light); text-transform: uppercase; }
.season-card .detail { font-size: 13px; color: var(--text-light); margin-top: 8px; }
.report-footer { background: var(--primary); color: white; padding: 40px; text-align: center; margin-top: 60px; }
.report-footer p { color: rgba(255,255,255,0.8); font-size: 13px; text-align: center; }
@media print { .cover { min-height: 100vh; } .figure-container { page-break-inside: avoid; } }
@media (max-width: 768px) { .cover h1 { font-size: 28px; } .season-grid { grid-template-columns: 1fr; } .key-figures { grid-template-columns: 1fr 1fr; } }
"""


def _fmt(n: int | float) -> str:
    """Format a number with commas."""
    return f'{int(n):,}'


def _fmt_short(n: int | float) -> str:
    """Format a number in short form (e.g., 6.2M, 350K)."""
    n = int(n)
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    elif n >= 1_000:
        return f'{n / 1_000:.0f}K'
    return str(n)


def _parse_number(s: str) -> int:
    """Parse a formatted number string like '6,227,072' to int."""
    try:
        return int(s.replace(',', '').strip())
    except (ValueError, AttributeError):
        return 0


def _figure_html(img_path: str, label: str, caption: str, fig_num: int, max_width: str = "100%") -> str:
    """Generate a figure container HTML block."""
    return f'''
    <div class="figure-container">
        <div class="figure-label">Figure {fig_num}</div>
        <img src="{escape(img_path)}" alt="{escape(label)}" style="max-width: {max_width};">
        <p class="figure-caption"><strong>Figure {fig_num}:</strong> {escape(caption)}</p>
    </div>'''


def _country_table_html(rows: list[dict]) -> str:
    """Generate an HTML table for country data."""
    if not rows:
        return '<p><em>No country data available.</em></p>'

    html = '''<table>
    <thead><tr>
        <th>Rank</th><th>Country</th>
        <th style="text-align:right">Displaced</th>
        <th style="text-align:right">% of Total</th>
        <th style="text-align:right">Events</th>
        <th>Primary Driver</th>
    </tr></thead><tbody>'''

    grand_total = sum(r['total'] for r in rows)
    for r in rows:
        hl = ' class="hl"' if r['rank'] <= 3 else ''
        html += f'''<tr{hl}>
            <td>{r['rank']}</td><td>{escape(r['country'])}</td>
            <td class="num">{_fmt(r['total'])}</td>
            <td class="num">{r['pct']:.1f}%</td>
            <td class="num">{r['events']}</td>
            <td>{escape(r['driver'])}</td>
        </tr>'''

    html += f'''<tr class="hl">
        <td colspan="2"><strong>TOTAL</strong></td>
        <td class="num"><strong>{_fmt(grand_total)}</strong></td>
        <td class="num"><strong>100%</strong></td>
        <td class="num"><strong>{sum(r['events'] for r in rows)}</strong></td>
        <td>&mdash;</td>
    </tr></tbody></table>'''
    return html


def _disaster_table_html(rows: list[dict]) -> str:
    """Generate an HTML table for disaster type data."""
    if not rows:
        return '<p><em>No disaster type data available.</em></p>'

    html = '''<table>
    <thead><tr>
        <th>Disaster Type</th>
        <th style="text-align:right">Displaced</th>
        <th style="text-align:right">% of Disaster Total</th>
        <th style="text-align:right">Events</th>
    </tr></thead><tbody>'''

    for i, r in enumerate(rows):
        hl = ' class="hl"' if i == 0 else ''
        html += f'''<tr{hl}>
            <td>{escape(r['type'])}</td>
            <td class="num">{_fmt(r['total'])}</td>
            <td class="num">{r['pct']:.1f}%</td>
            <td class="num">{r['events']}</td>
        </tr>'''

    html += '</tbody></table>'
    return html


# ============================================================
# SECTION BUILDERS
# ============================================================

def _build_cover(season_cfg: dict, year: int, stats: dict) -> str:
    """Build the cover page HTML."""
    season = season_cfg['name']
    is_annual = season.upper() == 'ANNUAL'
    title_label = "Full Year" if is_annual else season
    long_name = season_cfg['long_name']

    total_str = stats.get('total_displaced', '0')
    total_num = _parse_number(total_str)
    events_str = stats.get('num_events', '0')
    countries_str = stats.get('num_countries', '0')

    # Extract conflict percentage
    conflict_pct = ''
    for name, val in stats.get('type_breakdown', {}).items():
        if name == 'Conflict':
            conflict_pct = val.split('(')[1].rstrip(')') if '(' in val else ''

    if is_annual:
        subtitle = f"Annual Analytical Report &mdash; January to December {year}"
        description = (
            f"A comprehensive analysis of internal displacement patterns across Eastern Africa, "
            f"covering seasonal dynamics (MAM, JJAS, OND), disaster typologies, and country-level "
            f"impacts based on IDMC displacement data."
        )
    else:
        months_range = f"{season_cfg['months'][0]} to {season_cfg['months'][-1]}"
        subtitle = f"{season} Season Analytical Report &mdash; {months_range} {year}"
        description = (
            f"An analysis of internal displacement patterns across Eastern Africa during the "
            f"{season} ({long_name}) season, covering disaster typologies and country-level impacts."
        )

    cover_stats_html = f'''
    <div class="cover-stats">
        <div class="cover-stat"><div class="number">{_fmt_short(total_num)}</div><div class="desc">Total Displaced</div></div>
        <div class="cover-stat"><div class="number">{events_str}</div><div class="desc">Displacement Events</div></div>
        <div class="cover-stat"><div class="number">{countries_str}</div><div class="desc">Countries Affected</div></div>'''

    if conflict_pct:
        cover_stats_html += f'''
        <div class="cover-stat"><div class="number">{conflict_pct}</div><div class="desc">Conflict-Driven</div></div>'''

    cover_stats_html += '</div>'

    now = datetime.now()
    return f'''
    <div class="cover"><div class="cover-content">
        <div class="cover-badge">IGAD Region &bull; Analytical Report</div>
        <h1>Disaster and Conflict-Induced Internal Displacement in Eastern Africa</h1>
        <h2>{subtitle}</h2>
        <p style="font-size:15px; opacity:0.85; max-width:650px; margin: 0 auto 20px auto; text-align:center;">{description}</p>
        {cover_stats_html}
        <div class="cover-meta">
            <div class="cover-meta-item"><div class="label">Data Source</div><div class="value">IDMC / IDUS</div></div>
            <div class="cover-meta-item"><div class="label">Analysis Period</div><div class="value">{title_label} {year}</div></div>
            <div class="cover-meta-item"><div class="label">Produced By</div><div class="value">ICPAC Disaster Unit</div></div>
            <div class="cover-meta-item"><div class="label">Report Date</div><div class="value">{now.strftime("%B %Y")}</div></div>
        </div>
    </div></div>'''


def _build_executive_summary(stats: dict, country_rows: list[dict], disaster_rows: list[dict], season_cfg: dict) -> str:
    """Build the executive summary section."""
    season = season_cfg['name']
    is_annual = season.upper() == 'ANNUAL'
    title_label = "Full Year" if is_annual else season

    total_str = stats.get('total_displaced', 'N/A')
    total_num = _parse_number(total_str)
    events_str = stats.get('num_events', 'N/A')
    countries_str = stats.get('num_countries', 'N/A')
    top_country = stats.get('top_country', 'N/A')

    type_bd = stats.get('type_breakdown', {})
    conflict_info = type_bd.get('Conflict', 'N/A')
    disaster_info = type_bd.get('Disaster', 'N/A')

    # Parse numbers from type breakdown
    conflict_num = _parse_number(conflict_info.split('(')[0]) if conflict_info != 'N/A' else 0
    disaster_num = _parse_number(disaster_info.split('(')[0]) if disaster_info != 'N/A' else 0

    # Top 3 countries description
    top3_desc = ''
    if len(country_rows) >= 3:
        t = country_rows
        top3_total = sum(r['total'] for r in t[:3])
        top3_pct = (top3_total / total_num * 100) if total_num else 0
        top3_desc = (
            f'The three most affected countries &mdash; {t[0]["country"]}, {t[1]["country"]}, '
            f'and {t[2]["country"]} &mdash; together accounted for <strong>{top3_pct:.1f}%</strong> '
            f'of all displacements ({_fmt(top3_total)} out of {_fmt(total_num)}).'
        )

    # Top disaster type
    top_disaster_desc = ''
    if disaster_rows:
        d = disaster_rows[0]
        top_disaster_desc = (
            f'Among natural disasters, <strong>{d["type"]}</strong> was the most destructive hazard, '
            f'causing <strong>{_fmt(d["total"])} displacements</strong> ({d["pct"]:.1f}% of all disaster-related displacements).'
        )

    return f'''
    <section>
        <h2 class="section-title">1. Executive Summary</h2>
        <p>During the {title_label} {stats.get("_year", "")} analysis period, a total of
        <strong>{total_str} internally displaced persons (IDPs)</strong> were recorded across
        <strong>{countries_str} countries</strong> through <strong>{events_str} distinct displacement events</strong>
        in Eastern Africa.</p>

        <div class="key-figures">
            <div class="key-figure danger"><div class="number">{_fmt_short(total_num)}</div><div class="label">Total Displaced</div></div>
            <div class="key-figure warning"><div class="number">{_fmt_short(conflict_num)}</div><div class="label">Conflict Displaced</div></div>
            <div class="key-figure"><div class="number">{_fmt_short(disaster_num)}</div><div class="label">Disaster Displaced</div></div>
            <div class="key-figure success"><div class="number">{events_str}</div><div class="label">Events</div></div>
        </div>

        <p><strong>Conflict</strong> accounted for {conflict_info} of displacements, while <strong>Disaster</strong>
        accounted for {disaster_info}. The top affected country was <strong>{top_country}</strong>.</p>

        {"<div class='callout alert'><strong>Key Finding:</strong> " + top3_desc + "</div>" if top3_desc else ""}
        {"<p>" + top_disaster_desc + "</p>" if top_disaster_desc else ""}
    </section>'''


def _build_methodology(season_cfg: dict, year: int) -> str:
    """Build the methodology section."""
    return f'''
    <section>
        <h2 class="section-title">2. Data Sources and Methodology</h2>
        <h3 class="sub-title">2.1 Data Source</h3>
        <p>This analysis is based on data from the <strong>Internal Displacement Monitoring Centre (IDMC)</strong>
        Internal Displacement Updates (IDUS) database, accessed via the IDMC API. The dataset covers internal
        displacement events in 11 Eastern African countries within the IGAD region: <strong>Burundi, Djibouti,
        Eritrea, Ethiopia, Kenya, Rwanda, Somalia, South Sudan, Sudan, Tanzania,</strong> and <strong>Uganda</strong>.
        Data was sourced from: {DATA_SOURCE_ORGANIZATIONS}, IGAD-TAC, and ReliefWeb.</p>

        <h3 class="sub-title">2.2 Seasonal Framework</h3>
        <table>
            <thead><tr><th>Season</th><th>Months</th><th>Climatic Significance</th></tr></thead>
            <tbody>
                <tr><td><strong>MAM</strong></td><td>March, April, May</td><td>Long rains season; first major flood risk period</td></tr>
                <tr><td><strong>JJAS</strong></td><td>June, July, August, September</td><td>Kiremt/main rainy season for Ethiopia and northern tier</td></tr>
                <tr><td><strong>OND</strong></td><td>October, November, December</td><td>Short rains (Deyr) season; second flood risk period</td></tr>
            </tbody>
        </table>
        <p>Displacement figures are categorized by <strong>type</strong> (Conflict vs. Disaster) and by
        <strong>disaster sub-type</strong> (Flood, Drought, Earthquake, Storm, Mass Movement, Wildfire).</p>
    </section>'''


def _build_analysis_section(
    stats: dict, country_rows: list[dict], disaster_rows: list[dict],
    images: dict, season_cfg: dict, year: int,
) -> str:
    """Build the main analysis section with figures and tables."""
    season = season_cfg['name']
    is_annual = season.upper() == 'ANNUAL'
    title_label = "Full Year" if is_annual else season
    fig_num = 1
    html = f'''<section>
        <h2 class="section-title">3. {title_label} {year} Displacement Analysis</h2>'''

    # 3.1 Geographic Distribution
    html += f'<h3 class="sub-title">3.1 Geographic Distribution of Displacements</h3>'
    if 'displacements_by_country' in images:
        html += _figure_html(
            images['displacements_by_country'],
            f'{title_label} {year} Displacements by Country',
            f'Internal displacements by country for {title_label} {year}. '
            f'The chart shows the distribution of displacement across affected countries.',
            fig_num
        )
        fig_num += 1

    html += _country_table_html(country_rows)

    # 3.2 Displacement by Type
    html += f'<h3 class="sub-title">3.2 Displacement by Type: Conflict vs. Disaster</h3>'
    if 'displacement_by_type_pie' in images:
        html += _figure_html(
            images['displacement_by_type_pie'],
            f'{title_label} {year} Displacement by Type',
            f'Proportional breakdown of total displacements by type (Conflict vs. Disaster) for {title_label} {year}.',
            fig_num, max_width="550px"
        )
        fig_num += 1

    type_bd = stats.get('type_breakdown', {})
    if type_bd:
        html += '<p>'
        for name, val in type_bd.items():
            html += f'<strong>{escape(name)}</strong>: {escape(val)}. '
        html += '</p>'

    # 3.3 Disaster Type Breakdown
    html += f'<h3 class="sub-title">3.3 Disaster-Induced Displacement by Hazard Type</h3>'
    if 'displacements_by_disaster_type' in images:
        html += _figure_html(
            images['displacements_by_disaster_type'],
            f'{title_label} {year} Displacements by Disaster Type',
            f'Disaster-induced displacements by hazard type for {title_label} {year}.',
            fig_num
        )
        fig_num += 1

    html += _disaster_table_html(disaster_rows)

    # 3.4 Disaster Map
    html += f'<h3 class="sub-title">3.4 Spatial Distribution of Disaster Types</h3>'
    if 'disaster_map_with_donuts' in images:
        html += _figure_html(
            images['disaster_map_with_donuts'],
            f'Disaster Types in Eastern Africa - {title_label} {year}',
            f'Spatial distribution of disaster types across Eastern Africa ({title_label} {year}). '
            f'Icons represent disaster types reported per country. Donut charts summarize '
            f'total events, total displaced, and total affected.',
            fig_num
        )
        fig_num += 1

    # 3.5 Monthly Trend
    html += f'<h3 class="sub-title">3.5 Monthly Displacement Trend</h3>'
    if 'monthly_trend' in images:
        html += _figure_html(
            images['monthly_trend'],
            f'Monthly Displacement Trend - {year}',
            f'Monthly displacement trend for {title_label} {year}, showing temporal patterns '
            f'in displacement activity across the analysis period.',
            fig_num
        )
        fig_num += 1

    # Monthly breakdown table
    monthly_bd = stats.get('monthly_breakdown', {})
    if monthly_bd:
        html += '''<table><thead><tr><th>Month</th><th style="text-align:right">Displaced</th>
            <th style="text-align:right">% of Total</th></tr></thead><tbody>'''
        for month, val in monthly_bd.items():
            # val is like "352,561 (5.7%)"
            parts = val.split('(')
            displaced = parts[0].strip()
            pct = parts[1].rstrip(')').strip() if len(parts) > 1 else ''
            html += f'<tr><td>{escape(month)}</td><td class="num">{escape(displaced)}</td><td class="num">{escape(pct)}</td></tr>'
        html += '</tbody></table>'

    # 3.6 Heatmap
    html += f'<h3 class="sub-title">3.6 Country-Month Displacement Heatmap</h3>'
    if 'country_month_heatmap' in images:
        html += _figure_html(
            images['country_month_heatmap'],
            f'{title_label} {year} Country vs Month Heatmap',
            f'Country-by-month displacement heatmap for {title_label} {year}. '
            f'Darker shading indicates higher displacement intensity, enabling '
            f'identification of spatial-temporal hotspots.',
            fig_num
        )
        fig_num += 1

    html += '</section>'
    return html


def _build_seasonal_comparison(
    output_dir: Path, year: int, seasons: list[str],
) -> str:
    """Build a cross-season comparison section (for ANNUAL reports)."""
    season_data = []
    for s in seasons:
        cfg = get_season_config(s)
        prefix = s.lower()
        st = _load_summary_stats(output_dir, prefix)
        if not st:
            continue
        imgs = _get_season_images(output_dir, prefix, year)
        st['_season'] = s
        st['_cfg'] = cfg
        st['_images'] = imgs
        season_data.append(st)

    if not season_data:
        return ''

    html = '''<section>
        <h2 class="section-title">4. Seasonal Comparison</h2>
        <p>Eastern Africa's displacement patterns are strongly influenced by seasonal rainfall cycles.
        This section compares displacement characteristics across the three primary seasons.</p>
        <div class="season-grid">'''

    for sd in season_data:
        s = sd['_season']
        cfg = sd['_cfg']
        total_num = _parse_number(sd.get('total_displaced', '0'))
        events_str = sd.get('num_events', '0')
        countries_str = sd.get('num_countries', '0')
        type_bd = sd.get('type_breakdown', {})
        conflict_info = type_bd.get('Conflict', '')
        conflict_pct = ''
        if '(' in conflict_info:
            conflict_pct = conflict_info.split('(')[1].rstrip(')')

        months_str = f"{cfg['months'][0][:3]}&ndash;{cfg['months'][-1][:3]}"
        html += f'''
        <div class="season-card">
            <h4>{s} Season</h4>
            <div class="stat-label">{months_str}</div>
            <div class="stat">{_fmt_short(total_num)}</div>
            <div class="stat-label">Displaced</div>
            <div class="detail">{events_str} events &bull; {countries_str} countries</div>
            {"<div class='detail' style='color:var(--danger);'>" + conflict_pct + " Conflict</div>" if conflict_pct else ""}
        </div>'''

    html += '</div>'

    # Comparison table
    html += '''<table><thead><tr><th>Metric</th>'''
    for sd in season_data:
        html += f'<th style="text-align:right">{sd["_season"]}</th>'
    html += '</tr></thead><tbody>'

    metrics = [
        ('Total Displaced', 'total_displaced'),
        ('Number of Events', 'num_events'),
        ('Countries Affected', 'num_countries'),
        ('Top Country', 'top_country'),
    ]
    for label, key in metrics:
        html += f'<tr><td><strong>{label}</strong></td>'
        for sd in season_data:
            html += f'<td class="num">{escape(sd.get(key, "N/A"))}</td>'
        html += '</tr>'

    # Type breakdown rows
    for dtype in ['Conflict', 'Disaster']:
        html += f'<tr><td><strong>{dtype} Share</strong></td>'
        for sd in season_data:
            val = sd.get('type_breakdown', {}).get(dtype, 'N/A')
            pct = ''
            if '(' in val:
                pct = val.split('(')[1].rstrip(')')
            html += f'<td class="num">{escape(pct) if pct else escape(val)}</td>'
        html += '</tr>'

    html += '</tbody></table>'

    # Individual season maps
    for sd in season_data:
        s = sd['_season']
        cfg = sd['_cfg']
        imgs = sd['_images']
        long_name = cfg['long_name']

        html += f'<h3 class="sub-title">4.{season_data.index(sd)+1} {s} Season ({long_name})</h3>'

        total_num = _parse_number(sd.get('total_displaced', '0'))
        events_str = sd.get('num_events', '0')
        top_country = sd.get('top_country', 'N/A')

        html += f'''<p>The {s} season recorded <strong>{sd.get("total_displaced", "N/A")} displacements</strong>
        across {events_str} events. The top affected country was <strong>{escape(top_country)}</strong>.</p>'''

        if 'disaster_map_with_donuts' in imgs:
            html += _figure_html(
                imgs['disaster_map_with_donuts'],
                f'Disaster Types - {s} {sd.get("_year", "")}',
                f'Spatial distribution of disaster types during the {s} season ({long_name}).',
                0  # Will be labeled by season name instead
            ).replace('Figure 0', f'{s} Map')

    html += '</section>'
    return html


def _build_data_sources() -> str:
    """Build the data sources / annex section."""
    return f'''
    <section>
        <h2 class="section-title">Data Sources and Acknowledgements</h2>
        <table>
            <thead><tr><th>Source</th><th>Description</th></tr></thead>
            <tbody>
                <tr><td><strong>IDMC</strong></td><td>Internal Displacement Monitoring Centre &ndash; IDUS database (primary)</td></tr>
                <tr><td>ECHO</td><td>European Civil Protection and Humanitarian Aid Operations</td></tr>
                <tr><td>FEWS NET</td><td>Famine Early Warning Systems Network</td></tr>
                <tr><td>IOM</td><td>International Organization for Migration &ndash; Displacement Tracking Matrix</td></tr>
                <tr><td>UNOCHA</td><td>United Nations Office for the Coordination of Humanitarian Affairs</td></tr>
                <tr><td>IPC</td><td>Integrated Food Security Phase Classification</td></tr>
                <tr><td>IFRC</td><td>International Federation of Red Cross and Red Crescent Societies</td></tr>
                <tr><td>WHO</td><td>World Health Organization</td></tr>
                <tr><td>IGAD-TAC</td><td>IGAD Technical Advisory Committee</td></tr>
                <tr><td>ReliefWeb</td><td>OCHA information service for humanitarian crises</td></tr>
            </tbody>
        </table>
        <p><strong>Geographic Data:</strong> ICPAC Administrative Level 0 Boundaries. Humanitarian icons based on OCHA Visual Identity standards.</p>
        <p><strong>Note:</strong> Displacement figures represent internal displacements (flows), not cumulative IDP stocks.
        Figures are sourced from multiple humanitarian organizations and may include estimates.
        The boundaries and names used on maps do not imply official endorsement.</p>
    </section>'''


def _build_footer(season_cfg: dict, year: int) -> str:
    """Build the report footer."""
    now = datetime.now()
    return f'''
    <div class="report-footer">
        <p><strong>ICPAC &mdash; Climate Prediction and Applications Centre</strong></p>
        <p>Disaster and Conflict-Induced Internal Displacement in Eastern Africa &mdash; {season_cfg['name']} {year}</p>
        <p style="margin-top: 10px; font-size: 12px;">Data sources: {DATA_SOURCE_ORGANIZATIONS}, IGAD-TAC, ReliefWeb</p>
        <p style="margin-top: 5px; font-size: 11px; opacity: 0.6;">Report generated: {now.strftime("%B %Y")}</p>
    </div>'''


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_report(
    season: str,
    year: int,
    output_dir: Path = None,
    report_dir: Path = None,
) -> Path:
    """
    Generate an HTML displacement analysis report.

    Args:
        season: Season code (ANNUAL, MAM, JJAS, OND).
        year: Analysis year.
        output_dir: Directory containing images and summary files.
        report_dir: Directory to write the HTML report to.

    Returns:
        Path to the generated HTML report.
    """
    output_dir = output_dir or OUTPUT_DIR
    report_dir = report_dir or PROJECT_ROOT / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    season_cfg = get_season_config(season)
    prefix = season.lower()
    is_annual = season.upper() == 'ANNUAL'

    print(f"\nGenerating {season} {year} HTML report...")

    # Load data
    stats = _load_summary_stats(output_dir, prefix)
    stats['_year'] = str(year)
    df = _load_summary_excel(output_dir, prefix)
    images = _get_season_images(output_dir, prefix, year)
    country_rows = _compute_country_table(df)
    disaster_rows = _compute_disaster_table(df)

    print(f"  Stats loaded: {bool(stats.get('total_displaced'))}")
    print(f"  Excel rows: {len(df)}")
    print(f"  Images found: {len(images)} ({', '.join(images.keys())})")

    # Build HTML sections
    sections = []
    sections.append(_build_cover(season_cfg, year, stats))
    sections.append('<div class="container">')
    sections.append(_build_executive_summary(stats, country_rows, disaster_rows, season_cfg))
    sections.append(_build_methodology(season_cfg, year))
    sections.append(_build_analysis_section(stats, country_rows, disaster_rows, images, season_cfg, year))

    # For ANNUAL reports, add seasonal comparison
    if is_annual:
        seasonal_section = _build_seasonal_comparison(output_dir, year, ['MAM', 'JJAS', 'OND'])
        if seasonal_section:
            sections.append(seasonal_section)

    sections.append(_build_data_sources())
    sections.append('</div>')
    sections.append(_build_footer(season_cfg, year))

    # Assemble full HTML
    title_label = "Full Year" if is_annual else season
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Displacement Report - {title_label} {year} - Eastern Africa</title>
    <style>{_CSS}</style>
</head>
<body>
{''.join(sections)}
</body>
</html>'''

    # Write report - images referenced via relative path to output/
    report_filename = f'{prefix}_displacement_report_{year}.html'
    report_path = report_dir / report_filename
    report_path.write_text(html)

    print(f"  Report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML Displacement Analysis Report",
    )
    parser.add_argument('--season', required=True, choices=['OND', 'MAM', 'JJAS', 'ANNUAL'],
                        help='Season to generate report for')
    parser.add_argument('--year', type=int, required=True, help='Year')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory containing images/stats (default: output/)')
    parser.add_argument('--report-dir', type=Path, default=None,
                        help='Directory to write report to (default: reports/)')
    args = parser.parse_args()

    generate_report(args.season, args.year, args.output_dir, args.report_dir)


if __name__ == '__main__':
    main()
