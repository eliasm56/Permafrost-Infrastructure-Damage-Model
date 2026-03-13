#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot uncertainty + sensitivity (OAT) figures from BC damage model outputs (NO MAPS).

4-panel figure:
    Panels = {North America, Russia} × {Pooled across GCMs, Within-GCM avg}
    Bars   = modes, grouped by scenario (SSP2-4.5 vs SSP5-8.5)
    Metric = p95–p05 width (normalized by ALL within each scenario)

"""

import os
import glob
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# =========================================================
# CONFIG
# =========================================================
CSV_DIR = r"D:\PhD_main\chapter_2\outputs\damage_model_w_uncertainty"
OUT_FIG_DIR = r"/figs/mc_uncertainty"

# Which metric exists in *_MC_country_draws.csv (TOTAL)
DRAW_METRIC = "dam_cost_usd2024"   # or "dam_area_m2"

# Optional occupancy-split metrics in *_MC_country_draws.csv (present in ALL in your case)
RES_METRIC = "dam_cost_res_usd2024"
NONRES_METRIC = "dam_cost_nonres_usd2024"

# Macro groups
NA_COUNTRIES = {"USA", "CAN"}
RUS_COUNTRIES = {"RUS"}

# Friendly labels (SHORT names)
MODE_LABELS_SHORT = {
    "ALL": "All",
    "FS_ONLY": "Safety factor",
    "EXTENT_ONLY": "Damage extent",
    "DETECTION_ONLY": "Detection",
    "TYPE_ONLY": "Type labels",
    "STORIES_ONLY": "Stories",
}
MODE_ORDER = ["ALL", "FS_ONLY", "EXTENT_ONLY", "DETECTION_ONLY", "TYPE_ONLY", "STORIES_ONLY"]

MACRO_LABELS_SHORT = {
    "North America": "N. America",
    "Russia": "Russia",
}

SCENARIOS = ["SSP245", "SSP585"]
SCENARIO_TITLES = {
    "SSP245": "SSP2-4.5",
    "SSP585": "SSP5-8.5",
}

# Human-friendly hazard-map labels
CUSTOM_TITLES = {
    "bc-diff_ssp585_NorESM2-MM_2055-2064-2015-2024_nomask.nc": "NorESM2-MM",
    "bc-diff_ssp585_MPI-ESM1-2-HR_2055-2064-2015-2024_nomask.nc": "MPI-ESM1-2-HR",
    "bc-diff_ssp585_AWI-CM-1-1-MR_2055-2064-2015-2024_nomask.nc": "AWI-CM-1-1-MR",
    "bc_diff_ssp585_CESM2-WACCM_2055_2064-2015_2024.nc": "CESM2-WACCM",
    "bc-diff_ssp245_NorESM2-MM_2055-2064-2015-2024_nomask.nc": "NorESM2-MM",
    "bc-diff_ssp245_MPI-ESM1-2-HR_2055-2064-2015-2024_nomask.nc": "MPI-ESM1-2-HR",
    "bc-diff_ssp245_AWI-CM-1-1-MR_2055-2064-2015-2024_nomask.nc": "AWI-CM-1-1-MR",
    "bc_diff_ssp245_CESM2-WACCM_2055_2064-2015_2024.nc": "CESM2-WACCM",
}

# Regex patterns for your filenames:
MODE_REGEX_DRAWS_COUNTRY = re.compile(r"__(?P<mode>[A-Z0-9_]+)_MC_country_draws\.csv$", re.IGNORECASE)
MODE_REGEX_DRAWS_REGION  = re.compile(r"__(?P<mode>[A-Z0-9_]+)_MC_region_draws\.csv$",  re.IGNORECASE)

# Figure style controls
DPI = 1000
FONT_SIZE = 15
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})

# --- suptitle spacing controls (more space between title and plots) ---
SUPTITLE_Y = 0.985          # push suptitle upward
TIGHT_TOP  = 0.995          # reserve top space for suptitle
TIGHT_TOP_WITH_LEGEND = 0.88  # a bit more headroom when legend is outside/above

def finalize_figure(fig, suptitle, outpath, top=TIGHT_TOP, y=SUPTITLE_Y, dpi=DPI, right=1.0):
    fig.suptitle(suptitle, y=y, fontsize=FONT_SIZE)
    fig.tight_layout(rect=[0, 0.02, right, top])
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# Output filenames (friendly)
FIG_TOTAL_VIOLIN = "Fig_Uncertainty_Total_Violins.png"
FIG_HAZARDONLY_VIOLIN = "Fig_Uncertainty_HazardOnly_Violins.png"
FIG_WITHIN_BANDS = "Fig_Uncertainty_WithinModel_Bands.png"

FIG_OAT_VIOLINS_GRID = "Fig_Sensitivity_OAT_Violins.png"

FIG_REGIONAL_OCC_NA = "Fig_Regional_Occupancy_Bars_NorthAmerica_FULLMCS.png"
FIG_REGIONAL_OCC_RUS = "Fig_Regional_Occupancy_Bars_Russia_FULLMCS.png"

# NEW output filenames (occupancy-split for main figs only)
FIG_TOTAL_VIOLIN_OCC = "Fig_Uncertainty_Total_Violins_OccSplit.png"
FIG_HAZARDONLY_VIOLIN_OCC = "Fig_Uncertainty_HazardOnly_Violins_OccSplit.png"
FIG_WITHIN_BANDS_OCC = "Fig_Uncertainty_WithinModel_Bands_OccSplit.png"

# NEW: single consolidated 4-panel OAT attribution figure (width-based)
FIG_OAT_ATTR_4PANEL_WIDTH = "Fig_Sensitivity_OAT_Attribution_4Panel_Width.png"

# Colors for occupancy-split (region × occupancy) — USED ONLY FOR BANDS FIGURE
OCC_COLORS = {
    ("North America", "Residential"): "tab:blue",
    ("North America", "Non-residential"): "tab:cyan",
    ("Russia", "Residential"): "tab:orange",
    ("Russia", "Non-residential"): "tab:red",
}

# =========================================================
# UTIL
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def money_fmt(x, pos=None):
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:.1f}T"
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{x:.0f}"

def require_cols(df, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {name}: {missing}")

def strip_cols(df):
    df.columns = df.columns.astype(str).str.strip()
    return df

def macro_group_from_country(shape_group):
    sg = str(shape_group).upper()
    if sg in NA_COUNTRIES:
        return "North America"
    if sg in RUS_COUNTRIES:
        return "Russia"
    return "Other"

def hazard_label(hazard_map: str) -> str:
    h = str(hazard_map)
    if h in CUSTOM_TITLES:
        return CUSTOM_TITLES[h]
    m = re.search(r"_(NorESM2-MM|MPI-ESM1-2-HR|AWI-CM-1-1-MR|CESM2-WACCM)_", h)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(h))[0]

def clean_and_filter_draws_country(df, name="draws CSV"):
    """
    IMPORTANT: Only requires the TOTAL metric, so OAT mode files do not need res/nonres columns.
    Occupancy columns are converted if present (used only for main figs with ALL).
    """
    df = strip_cols(df)
    req = ["scenario", "hazard_map", "draw", "shapeGroup", DRAW_METRIC]
    require_cols(df, req, name)

    df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()
    df["shapeGroup"] = df["shapeGroup"].astype(str).str.upper()

    keep = NA_COUNTRIES.union(RUS_COUNTRIES)
    df = df[df["shapeGroup"].isin(keep)].copy()

    df["macro"] = df["shapeGroup"].map(macro_group_from_country)
    df = df[df["macro"].isin(["North America", "Russia"])].copy()

    df["hazard_label"] = df["hazard_map"].map(hazard_label)

    df[DRAW_METRIC] = pd.to_numeric(df[DRAW_METRIC], errors="coerce")
    for c in [RES_METRIC, NONRES_METRIC]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[DRAW_METRIC, "scenario", "hazard_map", "draw", "macro"])
    return df

# ---------- Region draws helpers (unchanged) ----------
def _detect_occ_cols_region_draws(df):
    cols = [c.strip() for c in df.columns.astype(str).tolist()]
    preferred_pairs = [
        ("dam_cost_res_usd2024", "dam_cost_nonres_usd2024"),
        ("dam_cost_residential_usd2024", "dam_cost_nonresidential_usd2024"),
        ("dam_cost_res_2024", "dam_cost_nonres_2024"),
        ("dam_cost_residential_2024", "dam_cost_nonresidential_2024"),
        ("res_dam_cost_usd2024", "nonres_dam_cost_usd2024"),
    ]
    lower_map = {c.lower(): c for c in cols}
    for a, b in preferred_pairs:
        if a in lower_map and b in lower_map:
            return lower_map[a], lower_map[b]

    def find_one(patterns):
        for c in cols:
            cl = c.lower()
            if any(re.search(p, cl) for p in patterns):
                return c
        return None

    res = find_one([r"dam(_)?cost.*res(idential)?", r"res(idential)?.*dam(_)?cost"])
    non = find_one([r"dam(_)?cost.*non(_)?res", r"non(_)?res(idential)?.*dam(_)?cost"])
    if res and non:
        return res, non

    raise KeyError(
        "Could not detect residential/non-residential damaged-cost columns in MC_region_draws.csv.\n"
        "Expected something like: dam_cost_res_usd2024 and dam_cost_nonres_usd2024 (names can vary)."
    )

def clean_and_filter_draws_region(df, name="region draws CSV"):
    df = strip_cols(df)
    base_req = ["scenario", "hazard_map", "draw", "shapeGroup", "shapeName"]
    require_cols(df, base_req, name)

    df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()
    df["shapeGroup"] = df["shapeGroup"].astype(str).str.upper()
    df["shapeName"] = df["shapeName"].astype(str).str.strip()

    keep = NA_COUNTRIES.union(RUS_COUNTRIES)
    df = df[df["shapeGroup"].isin(keep)].copy()

    df["macro"] = df["shapeGroup"].map(macro_group_from_country)
    df = df[df["macro"].isin(["North America", "Russia"])].copy()

    df["hazard_label"] = df["hazard_map"].map(hazard_label)

    res_col, non_col = _detect_occ_cols_region_draws(df)
    df[res_col] = pd.to_numeric(df[res_col], errors="coerce")
    df[non_col] = pd.to_numeric(df[non_col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["scenario", "hazard_map", "draw", "macro", "shapeName", res_col, non_col]).copy()
    return df, res_col, non_col

# =========================================================
# DISCOVERY
# =========================================================
def discover_mc_country_draw_files(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "*MC_country_draws.csv")))
    if not files:
        raise RuntimeError(f"No *MC_country_draws.csv files found in: {csv_dir}")

    out = {}
    for fp in files:
        base = os.path.basename(fp)
        m = MODE_REGEX_DRAWS_COUNTRY.search(base)
        if not m:
            continue
        mode = m.group("mode").upper()
        out[mode] = fp

    if not out:
        raise RuntimeError(
            "Found MC_country_draws files, but none matched the __<MODE>_MC_country_draws.csv pattern.\n"
            "Expected example: BC_risk_analysis_v3_uncert_netcdf__ALL_MC_country_draws.csv"
        )
    return out

def discover_mc_region_draw_file_for_mode(csv_dir, mode):
    pat = os.path.join(csv_dir, f"*__{mode}_MC_region_draws.csv")
    hits = sorted(glob.glob(pat))
    if hits:
        return hits[0]
    files = sorted(glob.glob(os.path.join(csv_dir, "*MC_region_draws.csv")))
    for fp in files:
        base = os.path.basename(fp)
        m = MODE_REGEX_DRAWS_REGION.search(base)
        if m and m.group("mode").upper() == mode.upper():
            return fp
    return None

def ordered_modes_present(modes_present):
    mp = [m.upper() for m in modes_present]
    out = []
    for m in MODE_ORDER:
        if m in mp:
            out.append(m)
    for m in sorted(mp):
        if m not in out:
            out.append(m)
    return out

# =========================================================
# CORE AGGREGATIONS
# =========================================================
def pooled_totals_by_draw(df):
    return (
        df.groupby(["scenario", "hazard_map", "hazard_label", "draw", "macro"], dropna=False)[DRAW_METRIC]
          .sum()
          .reset_index()
    )

def pooled_totals_by_hazard_mean(df_draw_totals):
    return (
        df_draw_totals.groupby(["scenario", "hazard_map", "hazard_label", "macro"], dropna=False)[DRAW_METRIC]
                      .mean()
                      .reset_index()
    )

def pooled_totals_by_draw_occ_country(df):
    """
    Needs RES_METRIC and NONRES_METRIC in df (used only for ALL main figures).
    """
    g = (
        df.groupby(["scenario", "hazard_map", "hazard_label", "draw", "macro"], dropna=False)[[RES_METRIC, NONRES_METRIC]]
          .sum()
          .reset_index()
    )
    long = g.melt(
        id_vars=["scenario", "hazard_map", "hazard_label", "draw", "macro"],
        value_vars=[RES_METRIC, NONRES_METRIC],
        var_name="occ_raw",
        value_name="value",
    )
    long["occ"] = np.where(long["occ_raw"] == RES_METRIC, "Residential", "Non-residential")
    return long.drop(columns=["occ_raw"])

def pooled_totals_by_hazard_mean_occ(df_draw_totals_occ):
    return (
        df_draw_totals_occ
        .groupby(["scenario", "hazard_map", "hazard_label", "macro", "occ"], dropna=False)["value"]
        .mean()
        .reset_index()
    )

def uncertainty_width_p05_p95(values):
    p05 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    p50 = float(np.percentile(values, 50))
    return p05, p50, p95, (p95 - p05)

def compute_widths_per_mode(mode_to_df):
    rows = []
    for mode, df in mode_to_df.items():
        tot = pooled_totals_by_draw(df)
        for scen in SCENARIOS:
            for macro in ["North America", "Russia"]:
                sub = tot[(tot["scenario"] == scen) & (tot["macro"] == macro)]
                if sub.empty:
                    continue
                vals = sub[DRAW_METRIC].to_numpy(np.float64)
                p05, p50, p95, w = uncertainty_width_p05_p95(vals)
                rows.append({
                    "mode": mode,
                    "scenario": scen,
                    "macro": macro,
                    "p05": p05,
                    "p50": p50,
                    "p95": p95,
                    "width_p95_p05": w,
                })
    return pd.DataFrame(rows)

# =========================================================
# OAT WIDTHS for consolidated attribution
# =========================================================
def _metrics_from_values(values: np.ndarray):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(p05=np.nan, p50=np.nan, p95=np.nan, width=np.nan)
    p05 = float(np.percentile(v, 5))
    p50 = float(np.percentile(v, 50))
    p95 = float(np.percentile(v, 95))
    width = p95 - p05
    return dict(p05=p05, p50=p50, p95=p95, width=width)

def compute_oat_widths(mode_to_df):
    """
    Returns DF with:
      mode, scenario, macro, scope in {'pooled','within'}, width
    where:
      pooled  = pool draws across hazard maps
      within  = compute width per hazard_map then average widths across hazard maps
    """
    rows = []
    for mode, df in mode_to_df.items():
        tot = pooled_totals_by_draw(df)

        for scen in SCENARIOS:
            for macro in ["North America", "Russia"]:
                sub = tot[(tot["scenario"] == scen) & (tot["macro"] == macro)]
                if sub.empty:
                    continue

                # pooled
                pooled_vals = sub[DRAW_METRIC].to_numpy(np.float64)
                m_pooled = _metrics_from_values(pooled_vals)
                rows.append({
                    "mode": mode, "scenario": scen, "macro": macro, "scope": "pooled",
                    "width": m_pooled["width"]
                })

                # within hazard_map then avg
                widths = []
                for _, g in sub.groupby("hazard_map", dropna=False):
                    vals = g[DRAW_METRIC].to_numpy(np.float64)
                    widths.append(_metrics_from_values(vals)["width"])
                if widths:
                    rows.append({
                        "mode": mode, "scenario": scen, "macro": macro, "scope": "within",
                        "width": float(np.nanmean(widths))
                    })

    return pd.DataFrame(rows)

# =========================================================
# PLOTS
# =========================================================
def _set_common_y(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(money_fmt))
    ax.set_ylabel("Damages (2024 USD)")

def plot_total_uncertainty_violins_2panel(df_all, out_dir):
    tot = pooled_totals_by_draw(df_all)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=DPI)
    for ax, scen in zip(axes, SCENARIOS):
        sub = tot[tot["scenario"] == scen]
        data, labels = [], []
        for macro in ["North America", "Russia"]:
            vals = sub.loc[sub["macro"] == macro, DRAW_METRIC].to_numpy(np.float64)
            data.append(vals)
            labels.append(MACRO_LABELS_SHORT.get(macro, macro))

        ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, rotation=25)
        _set_common_y(ax)
        ax.set_title(SCENARIO_TITLES.get(scen, scen))

    outpath = os.path.join(out_dir, FIG_TOTAL_VIOLIN)
    finalize_figure(fig, "Uncertainty in estimated building damages", outpath, top=TIGHT_TOP, y=SUPTITLE_Y)

def plot_hazard_only_uncertainty_violins_2panel(df_all, out_dir):
    tot = pooled_totals_by_draw(df_all)
    hm = pooled_totals_by_hazard_mean(tot)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=DPI)
    for ax, scen in zip(axes, SCENARIOS):
        sub = hm[hm["scenario"] == scen]
        data, labels = [], []
        for macro in ["North America", "Russia"]:
            vals = sub.loc[sub["macro"] == macro, DRAW_METRIC].to_numpy(np.float64)
            data.append(vals)
            labels.append(MACRO_LABELS_SHORT.get(macro, macro))

        ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, rotation=25)
        _set_common_y(ax)
        ax.set_title(SCENARIO_TITLES.get(scen, scen))

    outpath = os.path.join(out_dir, FIG_HAZARDONLY_VIOLIN)
    finalize_figure(fig, "Uncertainty from climate-model hazard projections", outpath, top=TIGHT_TOP, y=SUPTITLE_Y)

def plot_within_hazard_map_bands_2panel(df_all, out_dir, label_fontsize=13):
    tot = pooled_totals_by_draw(df_all)

    def summarize_within(scen):
        sub = tot[tot["scenario"] == scen].copy()
        if sub.empty:
            return None

        def q(x, p):
            return np.percentile(x, p)

        summ = (
            sub.groupby(["hazard_map", "hazard_label", "macro"], dropna=False)[DRAW_METRIC]
               .agg(
                   p05=lambda x: q(x, 5),
                   p50=lambda x: q(x, 50),
                   p95=lambda x: q(x, 95),
                   mean="mean",
               )
               .reset_index()
        )

        ref = summ[summ["macro"] == "North America"].sort_values("mean", ascending=False)
        order = ref["hazard_label"].tolist()
        if not order:
            order = summ["hazard_label"].drop_duplicates().tolist()
        return summ, order

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=DPI)

    for ax, scen in zip(axes, SCENARIOS):
        out = summarize_within(scen)
        if out is None:
            ax.set_axis_off()
            continue

        summ, hazard_order = out
        y = np.arange(len(hazard_order))

        offsets = {"North America": -0.15, "Russia": +0.15}
        for macro in ["North America", "Russia"]:
            s = summ[summ["macro"] == macro].set_index("hazard_label").reindex(hazard_order)
            p05 = s["p05"].to_numpy(np.float64)
            p50 = s["p50"].to_numpy(np.float64)
            p95 = s["p95"].to_numpy(np.float64)

            yy = y + offsets[macro]
            ax.hlines(yy, p05, p95, linewidth=2)
            ax.scatter(p50, yy, s=18)

        ax.set_yticks(y)
        ax.set_yticklabels(hazard_order, fontsize=label_fontsize)
        ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
        ax.set_xlabel("Damages (2024 USD)")
        ax.set_title(SCENARIO_TITLES.get(scen, scen))
        ax.text(
            0.98, 0.02,
            f"{MACRO_LABELS_SHORT['North America']} / {MACRO_LABELS_SHORT['Russia']}",
            transform=ax.transAxes, ha="right", va="bottom"
        )

    outpath = os.path.join(out_dir, FIG_WITHIN_BANDS)
    finalize_figure(fig, "Monte Carlo uncertainty within each hazard projection", outpath, top=TIGHT_TOP, y=SUPTITLE_Y)

# ---------------- OCCUPANCY-SPLIT MAIN FIGURES (1–3) ----------------
def _occ_legend_handles():
    handles = []
    for (macro, occ), col in OCC_COLORS.items():
        label = f"{MACRO_LABELS_SHORT.get(macro, macro)} – {occ}"
        handles.append(Line2D([0], [0], color=col, lw=2.2, marker="o", markersize=5, label=label))
    return handles

def plot_total_uncertainty_violins_2panel_occ_country(df_all, out_dir):
    tot = pooled_totals_by_draw_occ_country(df_all)

    order_pairs = [
        ("North America", "Residential"),
        ("North America", "Non-residential"),
        ("Russia", "Residential"),
        ("Russia", "Non-residential"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=DPI)
    for ax, scen in zip(axes, SCENARIOS):
        sub = tot[tot["scenario"] == scen]
        data, labels = [], []
        for macro, occ in order_pairs:
            vals = sub.loc[(sub["macro"] == macro) & (sub["occ"] == occ), "value"].to_numpy(np.float64)
            data.append(vals)
            labels.append(f"{MACRO_LABELS_SHORT.get(macro, macro)}\n{occ}")

        ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25)
        _set_common_y(ax)
        ax.set_title(SCENARIO_TITLES.get(scen, scen))

    outpath = os.path.join(out_dir, FIG_TOTAL_VIOLIN_OCC)
    finalize_figure(fig, "Uncertainty in estimated building damages", outpath, top=TIGHT_TOP, y=SUPTITLE_Y)

def plot_hazard_only_uncertainty_violins_2panel_occ_country(df_all, out_dir):
    tot = pooled_totals_by_draw_occ_country(df_all)
    hm = pooled_totals_by_hazard_mean_occ(tot)

    order_pairs = [
        ("North America", "Residential"),
        ("North America", "Non-residential"),
        ("Russia", "Residential"),
        ("Russia", "Non-residential"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=DPI)
    for ax, scen in zip(axes, SCENARIOS):
        sub = hm[hm["scenario"] == scen]
        data, labels = [], []
        for macro, occ in order_pairs:
            vals = sub.loc[(sub["macro"] == macro) & (sub["occ"] == occ), "value"].to_numpy(np.float64)
            data.append(vals)
            labels.append(f"{MACRO_LABELS_SHORT.get(macro, macro)}\n{occ}")

        ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25)
        _set_common_y(ax)
        ax.set_title(SCENARIO_TITLES.get(scen, scen))

    outpath = os.path.join(out_dir, FIG_HAZARDONLY_VIOLIN_OCC)
    finalize_figure(fig, "Uncertainty from climate-model hazard projections", outpath, top=TIGHT_TOP, y=SUPTITLE_Y)

def plot_within_hazard_map_bands_2panel_occ_country(df_all, out_dir):
    """
    Colors lines by (region × occupancy) and places ONE shared legend outside the axes.
    """
    tot = pooled_totals_by_draw_occ_country(df_all)

    def summarize_within(scen):
        sub = tot[tot["scenario"] == scen].copy()
        if sub.empty:
            return None

        def q(x, p):
            return np.percentile(x, p)

        summ = (
            sub.groupby(["hazard_map", "hazard_label", "macro", "occ"], dropna=False)["value"]
               .agg(
                   p05=lambda x: q(x, 5),
                   p50=lambda x: q(x, 50),
                   p95=lambda x: q(x, 95),
                   mean="mean",
               )
               .reset_index()
        )

        ref = (
            summ[summ["macro"] == "North America"]
            .groupby(["hazard_label"], dropna=False)["mean"].sum()
            .sort_values(ascending=False)
        )
        order = ref.index.tolist()
        if not order:
            order = summ["hazard_label"].drop_duplicates().tolist()
        return summ, order

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=DPI)

    offsets = {
        ("North America", "Residential"): -0.24,
        ("North America", "Non-residential"): -0.08,
        ("Russia", "Residential"): +0.08,
        ("Russia", "Non-residential"): +0.24,
    }

    for ax, scen in zip(axes, SCENARIOS):
        out = summarize_within(scen)
        if out is None:
            ax.set_axis_off()
            continue

        summ, hazard_order = out
        y = np.arange(len(hazard_order))

        for macro in ["North America", "Russia"]:
            for occ in ["Residential", "Non-residential"]:
                col = OCC_COLORS[(macro, occ)]
                s = (
                    summ[(summ["macro"] == macro) & (summ["occ"] == occ)]
                    .set_index("hazard_label")
                    .reindex(hazard_order)
                )

                p05 = s["p05"].to_numpy(np.float64)
                p50 = s["p50"].to_numpy(np.float64)
                p95 = s["p95"].to_numpy(np.float64)

                yy = y + offsets[(macro, occ)]
                ax.hlines(yy, p05, p95, linewidth=2.2, color=col)
                ax.scatter(p50, yy, s=18, color=col, edgecolor="none", zorder=5)

        ax.set_yticks(y)
        ax.set_yticklabels(hazard_order)
        ax.tick_params(axis="y", labelsize=FONT_SIZE)
        ax.tick_params(axis="x", labelsize=FONT_SIZE, labelrotation=25)

        ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
        ax.set_xlabel("Damages (2024 USD)", fontsize=FONT_SIZE)
        ax.set_title(SCENARIO_TITLES.get(scen, scen), fontsize=FONT_SIZE)

    fig.legend(
        handles=_occ_legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONT_SIZE
    )

    outpath = os.path.join(out_dir, FIG_WITHIN_BANDS_OCC)
    finalize_figure(fig, "Within-GCM Monte Carlo uncertainty", outpath, top=0.995, y=0.985)

# ---------------- OAT FIGURES (TOTAL ONLY) ----------------
def plot_oat_violins_grid(mode_to_df, modes_ordered, out_dir):
    nrows = len(modes_ordered)
    ncols = len(SCENARIOS)

    fig_w = 11.5
    fig_h = max(6.0, 2.35 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=DPI)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[a] for a in axes])

    for r, mode in enumerate(modes_ordered):
        df = mode_to_df[mode]
        tot = pooled_totals_by_draw(df)

        row_label = MODE_LABELS_SHORT.get(mode, mode)
        for c, scen in enumerate(SCENARIOS):
            ax = axes[r, c]
            sub = tot[tot["scenario"] == scen]
            data, labels = [], []
            for macro in ["North America", "Russia"]:
                vals = sub.loc[sub["macro"] == macro, DRAW_METRIC].to_numpy(np.float64)
                data.append(vals)
                labels.append(MACRO_LABELS_SHORT.get(macro, macro))

            ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels, rotation=25)
            ax.yaxis.set_major_formatter(FuncFormatter(money_fmt))
            ax.set_ylabel(row_label if c == 0 else "")

            if r == 0:
                ax.set_title(SCENARIO_TITLES.get(scen, scen))

    outpath = os.path.join(out_dir, FIG_OAT_VIOLINS_GRID)
    finalize_figure(fig, "One-at-a-time sensitivity: uncertainty by source", outpath, top=TIGHT_TOP, y=SUPTITLE_Y)

# ---------------- NEW: ONE consolidated 4-panel OAT attribution (width) ----------------
def plot_oat_attribution_4panel_width(mode_to_df, modes_ordered, out_dir):
    """
    2×2 panels:
      rows = {North America, Russia}
      cols = {Pooled across GCMs, Within-GCM avg}
    Bars are grouped by scenario (SSP2-4.5, SSP5-8.5) for each mode.
    Values are normalized by ALL within each scenario (so ALL=1.0).
    Metric is p95–p05 width.
    """
    W = compute_oat_widths(mode_to_df)
    if W.empty:
        print("[WARN] No OAT widths computed; skipping 4-panel attribution figure.")
        return

    # Normalize by ALL per (scenario, macro, scope)
    base = (
        W[W["mode"].str.upper() == "ALL"][["scenario", "macro", "scope", "width"]]
        .rename(columns={"width": "width_all"})
    )
    W = W.merge(base, on=["scenario", "macro", "scope"], how="left", validate="m:1")
    W["norm"] = W["width"] / W["width_all"]

    W["mode"] = W["mode"].str.upper()
    modes = modes_ordered

    fig, axes = plt.subplots(2, 2, figsize=(11, 14.0), dpi=DPI, sharey=True)

    panels = [
        ("North America", "pooled", 0, 0, "Pooled across GCMs"),
        ("North America", "within", 0, 1, "Within-GCM"),
        ("Russia",        "pooled", 1, 0, "Pooled (across GCMs)"),
        ("Russia",        "within", 1, 1, "Within-GCM"),
    ]

    x = np.arange(len(modes))
    bar_w = 0.20
    scen_offsets = {"SSP245": -bar_w/2, "SSP585": +bar_w/2}

    for macro, scope, r, c, col_title in panels:
        ax = axes[r, c]
        sub = W[(W["macro"] == macro) & (W["scope"] == scope)].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        for scen in SCENARIOS:
            ss = sub[sub["scenario"] == scen].set_index("mode").reindex(modes)
            y = ss["norm"].to_numpy(np.float64)
            ax.bar(x + scen_offsets[scen], y, width=bar_w, label=SCENARIO_TITLES.get(scen, scen))

        ax.axhline(1.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS_SHORT.get(m, m) for m in modes], rotation=25, ha="right")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{MACRO_LABELS_SHORT.get(macro, macro)} | {col_title}")

        if c == 0:
            ax.set_ylabel("Normalized uncertainty (p95–p05 width)")

    # Create a single legend for the whole figure (outside)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=False
    )

    outpath = os.path.join(out_dir, FIG_OAT_ATTR_4PANEL_WIDTH)
    finalize_figure(fig, "One-at-a-time attribution of uncertainty sources", outpath, right=0.82)


# =========================================================
# REGIONAL OCCUPANCY BARS (FULL MCS) — unchanged
# =========================================================
def _regional_occ_full_mcs_stats(df_region, macro, res_col, non_col):
    sub = df_region[df_region["macro"] == macro].copy()
    if sub.empty:
        return pd.DataFrame(columns=["scenario", "shapeName", "occ", "p50", "vmin", "vmax"])

    g = (
        sub.groupby(["scenario", "hazard_map", "draw", "shapeName"], dropna=False)[[res_col, non_col]]
           .sum()
           .reset_index()
    )

    rows = []
    for scen in SCENARIOS:
        gg = g[g["scenario"] == scen]
        if gg.empty:
            continue
        for region, gr in gg.groupby("shapeName", dropna=False):
            vr = gr[res_col].to_numpy(np.float64)
            vn = gr[non_col].to_numpy(np.float64)

            rows.append({"scenario": scen, "shapeName": region, "occ": "res",
                         "p50": float(np.percentile(vr, 50)),
                         "vmin": float(np.min(vr)),
                         "vmax": float(np.max(vr))})
            rows.append({"scenario": scen, "shapeName": region, "occ": "non",
                         "p50": float(np.percentile(vn, 50)),
                         "vmin": float(np.min(vn)),
                         "vmax": float(np.max(vn))})

    return pd.DataFrame(rows)

def plot_regional_occupancy_bars_full_mcs(df_region, res_col, non_col, out_dir):
    def _plot_macro(macro, outname):
        stats = _regional_occ_full_mcs_stats(df_region, macro, res_col, non_col)
        if stats.empty:
            print(f"[WARN] No regional draws available for macro={macro}; skipping.")
            return

        tmp = (
            stats.pivot_table(index=["scenario", "shapeName"], columns="occ", values="p50", aggfunc="first")
                 .reset_index()
        )
        tmp["total"] = tmp.get("res", 0.0) + tmp.get("non", 0.0)
        order = (
            tmp.groupby("shapeName")["total"]
               .mean()
               .sort_values(ascending=False)
               .index.tolist()
        )

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=DPI)

        for ax, scen in zip(axes, SCENARIOS):
            s = stats[stats["scenario"] == scen].copy()
            if s.empty:
                ax.set_axis_off()
                continue

            regs = [r for r in order if r in set(s["shapeName"].tolist())]
            x = np.arange(len(regs))
            w = 0.38

            def _get(occ, col):
                ss = s[s["occ"] == occ].set_index("shapeName").reindex(regs)
                return ss[col].to_numpy(np.float64)

            y_res = _get("res", "p50")
            y_non = _get("non", "p50")

            lo_res = _get("res", "vmin")
            hi_res = _get("res", "vmax")
            lo_non = _get("non", "vmin")
            hi_non = _get("non", "vmax")

            ax.bar(x - w/2, y_res, width=w, label="Residential")
            ax.bar(x + w/2, y_non, width=w, label="Non-residential")

            ax.errorbar(
                x - w/2, y_res,
                yerr=[np.maximum(0.0, y_res - lo_res), np.maximum(0.0, hi_res - y_res)],
                fmt="none", ecolor="black", capsize=3, elinewidth=1.3, capthick=1.3, zorder=5
            )
            ax.errorbar(
                x + w/2, y_non,
                yerr=[np.maximum(0.0, y_non - lo_non), np.maximum(0.0, hi_non - y_non)],
                fmt="none", ecolor="black", capsize=3, elinewidth=1.3, capthick=1.3, zorder=5
            )

            ax.set_title(SCENARIO_TITLES.get(scen, scen))
            ax.set_xticks(x)
            ax.set_xticklabels(regs, rotation=35, ha="right")
            ax.yaxis.set_major_formatter(FuncFormatter(money_fmt))
            ax.set_ylabel("Damages (2024 USD)")
            ax.grid(True, axis="y", alpha=0.25)

        axes[0].legend(frameon=False, loc="upper left")

        macro_title = MACRO_LABELS_SHORT.get(macro, macro)
        outpath = os.path.join(out_dir, outname)
        finalize_figure(
            fig,
            f"Regional damages by occupancy (bars = pooled MCS median; whiskers = pooled MCS range) — {macro_title}",
            outpath,
            top=TIGHT_TOP,
            y=SUPTITLE_Y
        )

    _plot_macro("North America", FIG_REGIONAL_OCC_NA)
    _plot_macro("Russia", FIG_REGIONAL_OCC_RUS)

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUT_FIG_DIR)

    # --- Discover + load COUNTRY draws for all modes (OAT + main totals) ---
    draw_files = discover_mc_country_draw_files(CSV_DIR)

    mode_to_df_country = {}
    for mode, fp in draw_files.items():
        df = pd.read_csv(fp)
        df = clean_and_filter_draws_country(df, name=os.path.basename(fp))
        if df.empty:
            continue
        mode_to_df_country[mode.upper()] = df

    if not mode_to_df_country:
        raise RuntimeError("All mode draw files loaded empty after filtering (USA/CAN/RUS).")

    base_mode = "ALL" if "ALL" in mode_to_df_country else sorted(mode_to_df_country.keys())[0]
    df_all_country = mode_to_df_country[base_mode]
    print(f"[INFO] Using mode='{base_mode}' for combined uncertainty figures (country draws).")

    modes_ordered = ordered_modes_present(mode_to_df_country.keys())
    print(f"[INFO] Modes present (ordered): {modes_ordered}")

    # 1) Total uncertainty (2-panel) — TOTAL
    plot_total_uncertainty_violins_2panel(df_all_country, OUT_FIG_DIR)

    # 2) Hazard-only uncertainty (2-panel) — TOTAL
    # plot_hazard_only_uncertainty_violins_2panel(df_all_country, OUT_FIG_DIR)

    # 3) Within-hazard-map MC bands (2-panel) — TOTAL
    # plot_within_hazard_map_bands_2panel(df_all_country, OUT_FIG_DIR, label_fontsize=12)

    # 1–3b) Occupancy-split versions ONLY if base_mode has the columns
    have_occ = (RES_METRIC in df_all_country.columns) and (NONRES_METRIC in df_all_country.columns)
    if have_occ:
        print(f"[INFO] Occupancy columns found in mode='{base_mode}'; writing occ-split main figures.")
        plot_total_uncertainty_violins_2panel_occ_country(df_all_country, OUT_FIG_DIR)
        plot_hazard_only_uncertainty_violins_2panel_occ_country(df_all_country, OUT_FIG_DIR)
        plot_within_hazard_map_bands_2panel_occ_country(df_all_country, OUT_FIG_DIR)
    else:
        print(f"[WARN] No occupancy columns ({RES_METRIC}, {NONRES_METRIC}) in mode='{base_mode}'. Skipping occ-split main figures.")

    # 4) OAT violins grid — TOTAL ONLY
    plot_oat_violins_grid(mode_to_df_country, modes_ordered, OUT_FIG_DIR)

    # 5) single, consolidated 4-panel OAT attribution figure (width-based)
    plot_oat_attribution_4panel_width(mode_to_df_country, modes_ordered, OUT_FIG_DIR)

    # 6) Regional occupancy bars (FULL MCS) from REGION draws (if present)
    region_mode = "ALL" if ("ALL" in mode_to_df_country) else base_mode
    fp_region = discover_mc_region_draw_file_for_mode(CSV_DIR, region_mode)
    if fp_region is None:
        print(f"[WARN] No MC_region_draws.csv found for mode='{region_mode}'. Skipping regional occupancy bars.")
    else:
        df_region_raw = pd.read_csv(fp_region)
        df_region, res_col, non_col = clean_and_filter_draws_region(df_region_raw, name=os.path.basename(fp_region))
        print(f"[INFO] Regional draws loaded from mode='{region_mode}': {os.path.basename(fp_region)}")
        print(f"[INFO] Using regional occupancy columns: res='{res_col}', nonres='{non_col}'")
        # plot_regional_occupancy_bars_full_mcs(df_region, res_col, non_col, OUT_FIG_DIR)

    print(f"[SUCCESS] Figures written to: {OUT_FIG_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
