# -*- coding: utf-8 -*-
"""
4-panel Arctic map of regional amplification in MEDIAN damages from B1 -> B4.

  - FULL POOLING:
      For each (scenario, shapeGroup, shapeName), compute STAT directly from the
      fully pooled distribution across ALL rows (hazard_map × draw).

      STAT = percentile_{(hazard_map, draw)}(dam_cost)

Amplification metrics (per region + scenario):
  - pct_increase = (median_B4 - median_B1) / median_B1 * 100
  - abs_increase = (median_B4 - median_B1)  [million USD]

Figure layout (2x2):
  Row 1: % increase
  Row 2: absolute increase (million USD)
  Col 1: SSP245
  Col 2: SSP585

This version:
- Uses Jenks natural breaks (mapclassify) split at 0
- Adds region labels at polygon centroids
- Places legends in dedicated whitespace bands BELOW each row (no overlap)
- FIG_SIZE=(12,12), DPI=1000
"""

import os
import re
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.patches as mpatches
from shapely.geometry import LineString
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import mapclassify

# =========================
# CONFIG
# =========================
FONT_SIZE    = 15
FIG_SIZE     = (12, 12)
FIG_DPI      = 1000
EDGE_COLOR   = "black"
EDGE_LW      = 0.25

CMAP_PCT_NAME = "RdYlGn"
CMAP_ABS_NAME = "RdYlGn"

# Total desired number of classes per row (split between negative/positive as needed)
N_CLASSES_PCT = 4
N_CLASSES_ABS = 4

# Projection (Arctic LAEA)
TARGET_EPSG  = 3995

# Basemap styling (ADM0)
OCEAN_FACE   = "#e6f2ff"
LAND_FACE    = "#f0f0f0"
COAST_EDGE   = "dimgray"
COAST_LW     = 0.3
ADM0_PATH    = "data/geoBoundariesCGAZ_ADM0.shp"  # set None to disable

# Polar graticule
GRATICULE_ON         = True
PARALLEL_STEP_DEG    = 10
MERIDIAN_STEP_DEG    = 30
GRATICULE_LAT_MIN    = 45
GRATICULE_SAMPLE_DEG = 0.5
GRAT_COLOR           = "gray"
GRAT_LS              = "--"
GRAT_LW              = 0.6
GRAT_ALPHA           = 0.8
LABEL_MERIDIANS      = True
LABEL_PARALLELS      = True
LABEL_OFFSET         = 150000   # meters

# Optional: keep only these shapeGroups (set None to keep all)
KEEP_GROUPS: Optional[set] = {"USA", "CAN", "RUS"}

SCENARIO_ORDER  = ["SSP245", "SSP585"]
SCENARIO_TITLES = {"SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}

STAT = "p50"  # across pooled hazard_map × draw

# Absolute increase units
SCALE_FACTOR_USD = 1e-6  # USD -> million USD
UNIT_LABEL_ABS = "Increase in median damages (million USD)"
UNIT_LABEL_PCT = "Increase in median damages (%)"

# =========================
# I/O
# =========================
FILE_B1 = "outputs/BC_decomposition_v1__B1_OSM_2D_COMBINEDCOST_MC_region_draws.csv"
FILE_B4 = "outputs/BC_decomposition_v1__B4_ADD_FLOORAREA_MC_region_draws.csv"

REGIONS_SHP = "data/regional_boundaries.shp"

OUT_DIR = "figs/maps"
OUT_PNG = os.path.join(OUT_DIR, "4panel_regional_amplification_B1_to_B4_median.png")

COST_COL_PREFS = ["dam_cost_ppp2024", "dam_cost_usd2024", "dam_cost"]

# =========================
# Matplotlib sizing
# =========================
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})

# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

_ws = re.compile(r"\s+")

def _standardize_scenario(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(_ws, "", regex=True).str.upper()
    return out

def _scenario_order_from_present(present: list) -> list:
    present_u = [p.upper() for p in present]
    ordered = []
    for want in SCENARIO_ORDER:
        if want.upper() in present_u:
            ordered.append(want.upper())
    for p in present_u:
        if p not in ordered:
            ordered.append(p)
    return ordered

def _clean_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing required field in CSV: '{c}'. Available: {list(df.columns)}")
        ser = (
            df[c].astype(str)
            .str.replace(r"[,\s]", "", regex=True)
            .str.replace(r"^\$", "", regex=True)
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
        df[c] = pd.to_numeric(ser, errors="coerce")
    return df

def _compute_stat_from_series(x: pd.Series, stat: str) -> float:
    a = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if a.size == 0:
        return np.nan
    stat = stat.lower().strip()
    if stat == "p50":
        return float(np.percentile(a, 50))
    if stat == "p05":
        return float(np.percentile(a, 5))
    if stat == "p95":
        return float(np.percentile(a, 95))
    if stat == "mean":
        return float(np.mean(a))
    raise ValueError("STAT must be one of: 'p50', 'p05', 'p95', 'mean'.")

def _detect_cost_col(df: pd.DataFrame) -> str:
    for c in COST_COL_PREFS:
        if c in df.columns:
            return c
    maybe = [c for c in df.columns if "dam_cost" in c]
    if len(maybe) == 1:
        return maybe[0]
    raise KeyError(f"Could not detect a dam_cost column. Candidates: {maybe}")

def _build_region_stat_from_mc_draws(csv_path: str, stat: str) -> pd.DataFrame:
    """
    Returns scenario x shapeGroup x shapeName using FULL POOLING:

      STAT computed directly from the pooled distribution across all rows
      (hazard_map × draw) for each geography.

    i.e., no intermediate averaging within draw.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    need = {"scenario", "hazard_map", "draw", "shapeGroup", "shapeName"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    if KEEP_GROUPS is not None:
        df = df[df["shapeGroup"].isin(KEEP_GROUPS)].copy()

    df["scenario"] = _standardize_scenario(df["scenario"])
    df = df[df["scenario"].isin([s.upper() for s in SCENARIO_ORDER])].copy()

    cost_col = _detect_cost_col(df)
    df = _clean_numeric(df, [cost_col])

    agg = (
        df.groupby(["scenario", "shapeGroup", "shapeName"], dropna=False)[cost_col]
          .agg(lambda x: _compute_stat_from_series(x, stat))
          .reset_index()
          .rename(columns={cost_col: f"dam_cost_{stat}"})
    )
    return agg

def _load_regions(regions_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(regions_path):
        raise FileNotFoundError(f"Region polygons not found: {regions_path}")
    gdf = gpd.read_file(regions_path)
    gdf.columns = gdf.columns.str.strip()
    if "shapeName" not in gdf.columns:
        raise KeyError(f"Regions shapefile must contain 'shapeName'. Available: {list(gdf.columns)}")
    if KEEP_GROUPS is not None and "shapeGroup" in gdf.columns:
        gdf = gdf[gdf["shapeGroup"].isin(KEEP_GROUPS)].copy()
    return gdf

def _make_common_bounds(gdfs_p) -> tuple:
    b = None
    for g in gdfs_p:
        tb = g.total_bounds
        if b is None:
            b = tb
        else:
            b = np.array([min(b[0], tb[0]), min(b[1], tb[1]), max(b[2], tb[2]), max(b[3], tb[3])], dtype=float)
    xmin, ymin, xmax, ymax = b.tolist()
    padx = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    pady = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    return (xmin - padx, ymin - pady, xmax + padx, ymax + pady)

def _round_edge(v: float) -> float:
    av = abs(v)
    if av < 10:
        return float(round(v, 1))
    if av < 1000:
        return float(round(v, 0))
    if av < 100000:
        return float(round(v, -1))
    return float(round(v, -2))

def _rounded_edges(edges_raw: np.ndarray) -> np.ndarray:
    r = np.array([_round_edge(v) for v in edges_raw], dtype=float)
    for i in range(1, len(r)):
        if r[i] <= r[i - 1]:
            r[i] = r[i - 1] + (1e-6 if r[i - 1] != 0 else 1.0)
    return r

def _jenks_edges_one_sided(values: pd.Series, k: int) -> np.ndarray:
    """
    Jenks edges for values that are strictly one-sided (all >=0 or all <=0).
    Returns edges as a strictly increasing array (k+1 edges).
    """
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.array([0.0, 1.0], dtype=float)

    if s.nunique() <= 1:
        v = float(s.iloc[0])
        if v == 0:
            return np.linspace(0, 1.0, k + 1).astype(float)
        return np.linspace(float(s.min()), float(s.max()), k + 1).astype(float)

    classifier = mapclassify.NaturalBreaks(s.values, k=k)
    bins = classifier.bins.astype(float)

    edges = np.concatenate(([float(s.min())], bins))
    edges = np.maximum.accumulate(edges + np.linspace(0, 1e-12, len(edges)))
    return edges

def _split_zero_centered_edges(series: pd.Series, k_total: int) -> np.ndarray:
    """
    Build edges that NEVER cross zero by classifying negatives and positives separately,
    and inserting 0 as an explicit boundary.

    If data are one-sided (all <=0 or all >=0), we just return one-sided edges.
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.array([0.0, 1.0], dtype=float)

    neg = s[s < 0]
    pos = s[s > 0]

    has_neg = not neg.empty
    has_pos = not pos.empty

    # One-sided: no crossing possible
    if has_neg and not has_pos:
        k = min(k_total, max(1, neg.nunique()))
        return _jenks_edges_one_sided(neg, k=k)
    if has_pos and not has_neg:
        k = min(k_total, max(1, pos.nunique()))
        return _jenks_edges_one_sided(pos, k=k)

    # Two-sided: split bins between sides
    k_neg = max(1, k_total // 2)
    k_pos = max(1, k_total - k_neg)

    k_neg = min(k_neg, max(1, neg.nunique()))
    k_pos = min(k_pos, max(1, pos.nunique()))

    edges_neg = _jenks_edges_one_sided(neg, k=k_neg)   # increasing, negative
    edges_pos = _jenks_edges_one_sided(pos, k=k_pos)   # increasing, positive

    # Keep strictly negative edges and strictly positive edges; stitch with 0
    edges_neg = edges_neg[edges_neg < 0]
    edges_pos = edges_pos[edges_pos > 0]

    if edges_neg.size == 0:
        edges_neg = np.array([float(neg.min())], dtype=float)
    if edges_pos.size == 0:
        edges_pos = np.array([float(pos.min()), float(pos.max())], dtype=float)

    combined = np.concatenate([edges_neg, np.array([0.0], dtype=float), edges_pos])
    combined = np.maximum.accumulate(combined + np.linspace(0, 1e-12, len(combined)))
    return combined

def _legend_patches_from_edges(cmap, edges_lbl, unit_title):
    """
    Build legend patches from arbitrary edges (len = n_classes + 1).
    """
    n_classes = len(edges_lbl) - 1
    patches = []
    is_pct = unit_title.endswith("(%)")

    def fmt(v):
        return f"{v:,.2f}" if is_pct else f"{v:,.0f}"

    for i in range(n_classes):
        lo, hi = edges_lbl[i], edges_lbl[i + 1]
        color = cmap(i)

        if i == 0:
            label = f"≤ {fmt(hi)}"
        elif i == n_classes - 1:
            label = f"> {fmt(lo)}"
        else:
            label = f"{fmt(lo)} – {fmt(hi)}"

        patches.append(mpatches.Patch(color=color, label=label))
    return patches

def _deg_label(val):
    sign = "-" if val < 0 else ""
    return f"{sign}{abs(int(val))}°"

def _make_polar_graticule(lat_min=45,
                          parallel_step=10,
                          meridian_step=30,
                          sample_step=0.5) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    lons = np.arange(-180, 180 + 1e-9, meridian_step)
    lats = np.arange(lat_min, 90 + 1e-9, parallel_step)

    parallel_lines = []
    for lat in lats:
        lon_seq = np.arange(-180, 180 + 1e-9, sample_step)
        coords = np.column_stack([lon_seq, np.full_like(lon_seq, lat)])
        parallel_lines.append(LineString(coords))
    gdf_par = gpd.GeoDataFrame({"kind": "parallel", "lat": lats}, geometry=parallel_lines, crs="EPSG:4326")

    meridian_lines = []
    for lon in lons:
        lat_seq = np.arange(lat_min, 90 + 1e-9, sample_step)
        coords = np.column_stack([np.full_like(lat_seq, lon), lat_seq])
        meridian_lines.append(LineString(coords))
    gdf_mer = gpd.GeoDataFrame({"kind": "meridian", "lon": lons}, geometry=meridian_lines, crs="EPSG:4326")

    gdf_par = gdf_par.to_crs(epsg=TARGET_EPSG)
    gdf_mer = gdf_mer.to_crs(epsg=TARGET_EPSG)
    return gdf_par, gdf_mer

def _draw_graticule(ax, gdf_par, gdf_mer, bounds):
    if not GRATICULE_ON:
        return

    gdf_par.plot(ax=ax, color=GRAT_COLOR, linewidth=GRAT_LW, linestyle=GRAT_LS, alpha=GRAT_ALPHA, zorder=2)
    gdf_mer.plot(ax=ax, color=GRAT_COLOR, linewidth=GRAT_LW, linestyle=GRAT_LS, alpha=GRAT_ALPHA, zorder=2)

    xmin, ymin, xmax, ymax = bounds

    if LABEL_PARALLELS and "lat" in gdf_par.columns:
        for lat, geom in zip(gdf_par["lat"].values, gdf_par.geometry.values):
            try:
                xlab = xmax - LABEL_OFFSET
                _, y = geom.coords[-1]
                y = np.clip(y, ymin + LABEL_OFFSET * 0.25, ymax - LABEL_OFFSET * 0.25)
                ax.text(
                    xlab, y, _deg_label(lat),
                    ha="right", va="center", fontsize=FONT_SIZE * 0.8, color=GRAT_COLOR,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
                    zorder=5
                )
            except Exception:
                pass

    if LABEL_MERIDIANS and "lon" in gdf_mer.columns:
        for lon, geom in zip(gdf_mer["lon"].values, gdf_mer.geometry.values):
            try:
                x, _ = geom.coords[-1]
                x = np.clip(x, xmin + LABEL_OFFSET * 0.25, xmax - LABEL_OFFSET * 0.25)
                ylab = ymax - LABEL_OFFSET * 0.6
                ax.text(
                    x, ylab, _deg_label(lon),
                    ha="center", va="top", fontsize=FONT_SIZE * 0.8, color=GRAT_COLOR,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
                    zorder=5
                )
            except Exception:
                pass

def _plot_basemap(ax, extent_bounds):
    ax.set_facecolor(OCEAN_FACE)
    if ADM0_PATH is not None:
        if not os.path.exists(ADM0_PATH):
            raise FileNotFoundError(f"Basemap (ADM0) not found: {ADM0_PATH}")
        world = gpd.read_file(ADM0_PATH).to_crs(epsg=TARGET_EPSG)
        world.plot(ax=ax, facecolor=LAND_FACE, edgecolor=COAST_EDGE, linewidth=COAST_LW, zorder=1)
    else:
        ax.set_facecolor(LAND_FACE)

    xmin, ymin, xmax, ymax = extent_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def _add_region_labels(ax, gdf_p, name_col="shapeName"):
    """
    Region name labels at polygon centroids.
    Allows manual per-region offsets (in projected meters).
    """
    label_offsets = {
        "Murmansk":      (0, -200000),
        "Arkhangelsk":   (0, -120000),
    }

    cent = gdf_p.geometry.centroid

    for i, name in enumerate(gdf_p[name_col].astype(str).values):
        if name is None or name == "nan":
            continue

        x = cent.iloc[i].x
        y = cent.iloc[i].y

        dx, dy = label_offsets.get(name, (0, 0))
        x = x + dx
        y = y + dy

        ax.text(
            x, y, name,
            ha="center",
            va="center",
            fontsize=FONT_SIZE * 0.52,
            color="black",
            zorder=6,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")]
        )

def _plot_panel(ax, gdf_p, value_col, cmap_name, edges_raw, bounds, gdf_par, gdf_mer, title):
    """
    Diverging red->green ramp with explicit zero boundary:
      - bins < 0 map to the RED side
      - bins > 0 map to the GREEN side
    """
    n_classes = max(1, len(edges_raw) - 1)

    # Identify which bins are negative/positive based on their midpoints
    mids = (edges_raw[:-1] + edges_raw[1:]) / 2.0
    neg_mask = mids < 0
    pos_mask = mids > 0

    n_neg = int(neg_mask.sum())
    n_pos = int(pos_mask.sum())

    # Build a list of colors: reds for negative bins, greens for positive bins
    base = plt.get_cmap(cmap_name)
    colors = []

    # red side (0.0 .. 0.5 of RdYlGn is red->yellow)
    if n_neg > 0:
        # sample toward the red end (avoid pure yellow at 0.5)
        reds = base(np.linspace(0.05, 0.45, n_neg))
        colors.extend(list(reds))

    # if there is a zero bin (rare, only if an edge repeats), treat as neutral
    n_zero = n_classes - n_neg - n_pos
    if n_zero > 0:
        neutral = base(0.5)
        colors.extend([neutral] * n_zero)

    # green side (0.5 .. 1.0 of RdYlGn is yellow->green)
    if n_pos > 0:
        greens = base(np.linspace(0.55, 0.95, n_pos))
        colors.extend(list(greens))

    # Make ListedColormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors, name=f"{cmap_name}_split")
    norm = BoundaryNorm(edges_raw, cmap.N, clip=False)

    _plot_basemap(ax, bounds)
    _draw_graticule(ax, gdf_par, gdf_mer, bounds)
    ax.set_axis_off()

    gdf_p.plot(
        column=value_col, cmap=cmap, norm=norm, ax=ax,
        edgecolor=EDGE_COLOR, linewidth=EDGE_LW, legend=False,
        missing_kwds={"color": "lightgray", "edgecolor": EDGE_COLOR, "hatch": "///"},
        zorder=3
    )

    _add_region_labels(ax, gdf_p)
    ax.set_title(title)

# =========================
# Legend placement utilities
# =========================
def _reserve_legend_bands(fig, axes, inter_band=0.060, bottom_band=0.075):
    fig.canvas.draw()

    toprow_bottom = min(axes[0, 0].get_position().y0, axes[0, 1].get_position().y0)
    bottomrow_top = max(axes[1, 0].get_position().y1, axes[1, 1].get_position().y1)
    bottomrow_bottom = min(axes[1, 0].get_position().y0, axes[1, 1].get_position().y0)

    gap = toprow_bottom - bottomrow_top
    if gap < inter_band:
        shift = inter_band - gap
        for ax in (axes[1, 0], axes[1, 1]):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])

    fig.canvas.draw()
    bottomrow_bottom = min(axes[1, 0].get_position().y0, axes[1, 1].get_position().y0)
    if bottomrow_bottom < bottom_band:
        lift = (bottom_band - bottomrow_bottom) + 0.003
        for ax in axes.ravel():
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + lift, pos.width, pos.height])

    fig.canvas.draw()
    toprow_bottom = min(axes[0, 0].get_position().y0, axes[0, 1].get_position().y0)
    bottomrow_top = max(axes[1, 0].get_position().y1, axes[1, 1].get_position().y1)
    bottomrow_bottom = min(axes[1, 0].get_position().y0, axes[1, 1].get_position().y0)

    y_leg1 = bottomrow_top + (toprow_bottom - bottomrow_top) * 0.50
    y_leg2 = bottomrow_bottom * 0.45

    return y_leg1, y_leg2

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_DIR)

    # 1) Compute pooled medians for B1 and B4 (FULL POOLING)
    b1 = _build_region_stat_from_mc_draws(FILE_B1, stat=STAT).rename(columns={f"dam_cost_{STAT}": "median_B1"})
    b4 = _build_region_stat_from_mc_draws(FILE_B4, stat=STAT).rename(columns={f"dam_cost_{STAT}": "median_B4"})

    df = b1.merge(b4, on=["scenario", "shapeGroup", "shapeName"], how="inner")
    if df.empty:
        raise RuntimeError("No overlapping regions between B1 and B4 after merge.")

    # 2) Amplification metrics
    df["abs_increase_usd"] = df["median_B4"] - df["median_B1"]

    base = df["median_B1"].to_numpy(float)
    newv = df["median_B4"].to_numpy(float)
    pct = np.full_like(base, np.nan, dtype=float)
    m = np.isfinite(base) & (base > 0) & np.isfinite(newv)
    pct[m] = (newv[m] - base[m]) / base[m] * 100.0
    df["pct_increase"] = pct

    # 3) Load polygons and merge
    regions = _load_regions(REGIONS_SHP)

    scenarios_present = sorted(df["scenario"].dropna().unique().tolist())
    ordered = _scenario_order_from_present(scenarios_present)
    if len(ordered) < 2:
        raise ValueError(f"Need at least two scenarios present; found: {ordered}")
    scen_left, scen_right = ordered[0], ordered[1]

    left_vals  = df[df["scenario"] == scen_left].copy()
    right_vals = df[df["scenario"] == scen_right].copy()

    # NOTE: join on shapeName only (matches your current workflow)
    reg_left  = regions.merge(left_vals,  on=["shapeName"], how="left")
    reg_right = regions.merge(right_vals, on=["shapeName"], how="left")

    for g in (reg_left, reg_right):
        g["pct_increase"] = pd.to_numeric(g["pct_increase"], errors="coerce")
        g["abs_increase_musd"] = pd.to_numeric(g["abs_increase_usd"], errors="coerce") * SCALE_FACTOR_USD

    # 4) Project + bounds
    reg_left_p  = reg_left.to_crs(epsg=TARGET_EPSG)
    reg_right_p = reg_right.to_crs(epsg=TARGET_EPSG)
    bounds = _make_common_bounds([reg_left_p, reg_right_p])

    # 5) Graticule
    gdf_par, gdf_mer = _make_polar_graticule(
        lat_min=GRATICULE_LAT_MIN,
        parallel_step=PARALLEL_STEP_DEG,
        meridian_step=MERIDIAN_STEP_DEG,
        sample_step=GRATICULE_SAMPLE_DEG
    )

    # 6) Class breaks (consistent across scenarios within each row)
    #    Split at 0 so no class spans negative→positive.
    all_pct = pd.concat([reg_left_p["pct_increase"], reg_right_p["pct_increase"]], axis=0)
    all_abs = pd.concat([reg_left_p["abs_increase_musd"], reg_right_p["abs_increase_musd"]], axis=0)

    edges_pct_raw = _split_zero_centered_edges(all_pct, N_CLASSES_PCT)
    edges_pct_lbl = _rounded_edges(edges_pct_raw.copy())

    edges_abs_raw = _split_zero_centered_edges(all_abs, N_CLASSES_ABS)
    edges_abs_lbl = _rounded_edges(edges_abs_raw.copy())

    # 7) Plot panels
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)

    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.90,
        bottom=0.08,
        hspace=0.22,
        wspace=0.03
    )

    # Row 1: % increase
    _plot_panel(
        axes[0, 0], reg_left_p, "pct_increase",
        CMAP_PCT_NAME, edges_pct_raw,
        bounds, gdf_par, gdf_mer,
        SCENARIO_TITLES.get(scen_left, scen_left)
    )
    _plot_panel(
        axes[0, 1], reg_right_p, "pct_increase",
        CMAP_PCT_NAME, edges_pct_raw,
        bounds, gdf_par, gdf_mer,
        SCENARIO_TITLES.get(scen_right, scen_right)
    )

    # Row 2: absolute increase
    _plot_panel(
        axes[1, 0], reg_left_p, "abs_increase_musd",
        CMAP_ABS_NAME, edges_abs_raw,
        bounds, gdf_par, gdf_mer,
        ""
    )
    _plot_panel(
        axes[1, 1], reg_right_p, "abs_increase_musd",
        CMAP_ABS_NAME, edges_abs_raw,
        bounds, gdf_par, gdf_mer,
        ""
    )

    axes[0, 0].text(
        0.02, 0.95, "Percent increase",
        transform=axes[0, 0].transAxes,
        ha="left", va="top",
        fontsize=FONT_SIZE, weight="bold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        zorder=10
    )
    axes[1, 0].text(
        0.02, 0.95, "Absolute increase",
        transform=axes[1, 0].transAxes,
        ha="left", va="top",
        fontsize=FONT_SIZE, weight="bold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        zorder=10
    )

    from matplotlib.colors import ListedColormap

    # --- legend colormaps must match the panel logic ---
    def _make_split_listed_cmap(cmap_name, edges_lbl):
        n_classes = max(1, len(edges_lbl) - 1)
        mids = (edges_lbl[:-1] + edges_lbl[1:]) / 2.0
        n_neg = int((mids < 0).sum())
        n_pos = int((mids > 0).sum())
        n_zero = n_classes - n_neg - n_pos

        base = plt.get_cmap(cmap_name)
        colors = []
        if n_neg > 0:
            colors.extend(list(base(np.linspace(0.05, 0.45, n_neg))))
        if n_zero > 0:
            colors.extend([base(0.5)] * n_zero)
        if n_pos > 0:
            colors.extend(list(base(np.linspace(0.55, 0.95, n_pos))))
        return ListedColormap(colors, name=f"{cmap_name}_split")

    cmap_pct = _make_split_listed_cmap(CMAP_PCT_NAME, edges_pct_lbl)
    pct_patches = _legend_patches_from_edges(cmap_pct, edges_pct_lbl, UNIT_LABEL_PCT)

    cmap_abs = _make_split_listed_cmap(CMAP_ABS_NAME, edges_abs_lbl)
    abs_patches = _legend_patches_from_edges(cmap_abs, edges_abs_lbl, UNIT_LABEL_ABS)

    # Reserve whitespace bands and compute legend y-anchors
    y_leg1, y_leg2 = _reserve_legend_bands(fig, axes, inter_band=0.060, bottom_band=0.080)

    # ncol: keep reasonable if edges become > 8
    ncol_pct = min(len(pct_patches), 8)
    ncol_abs = min(len(abs_patches), 8)

    fig.legend(
        handles=pct_patches,
        loc="center",
        bbox_to_anchor=(0.5, y_leg1),
        ncol=ncol_pct,
        fontsize=FONT_SIZE * 0.9,
        title=UNIT_LABEL_PCT,
        frameon=False
    )

    fig.legend(
        handles=abs_patches,
        loc="center",
        bbox_to_anchor=(0.5, y_leg2),
        ncol=ncol_abs,
        fontsize=FONT_SIZE * 0.9,
        title=UNIT_LABEL_ABS,
        frameon=False
    )

    fig.suptitle(
        "Regional amplification of median building damages",
        fontsize=FONT_SIZE + 2,
        y=0.965
    )

    ensure_dir(os.path.dirname(OUT_PNG))
    fig.savefig(
        OUT_PNG,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.15
    )
    plt.close(fig)
    print(f"[SUCCESS] Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
