# -*- coding: utf-8 -*-
"""
Decomposition of estimated building damages based on incremental refinements to building stock exposure inventory.

- B1–B4 are read from *_MC_country_draws.csv and use FULL POOLING:
      value = median across ALL rows (hazard_map × draw pooled)
  grouped by (scenario, shapeGroup).
- B5 is read from ALL_MC_country_draws.csv with the same pooled-median logic
  (mode==ALL filter applied if present).

Y-axis: billions (divide by 1e9).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
OUT_DIR  = "outputs/decomposition_results/"
FIG_DIR  = "figs"

# Full model (ALL) MC draws (country totals)
FULL_ALL_MC_COUNTRY = "damage_results/BC_risk_analysis_v3_uncert_netcdf__ALL_MC_country_draws.csv"

# =========================
# INPUT FILES FOR BLOCKS (MC COUNTRY DRAWS)
# =========================
B1_COUNTRY = os.path.join(OUT_DIR, "BC_decomposition_v1__B1_OSM_2D_COMBINEDCOST_MC_country_draws.csv")
B2_COUNTRY = os.path.join(OUT_DIR, "BC_decomposition_v1__B2_OSM_HABITAT_2D_COMBINEDCOST_MC_country_draws.csv")
B3_COUNTRY = os.path.join(OUT_DIR, "BC_decomposition_v1__B3_ADD_OCCUPANCY_COSTS_MC_country_draws.csv")
B4_COUNTRY = os.path.join(OUT_DIR, "BC_decomposition_v1__B4_ADD_FLOORAREA_MC_country_draws.csv")

BLOCK_FILES = {
    "B1": B1_COUNTRY,
    "B2": B2_COUNTRY,
    "B3": B3_COUNTRY,
    "B4": B4_COUNTRY,
}

BLOCK_LABELS = [
    "(OSM 2D footprint area, \ncombined replacement costs)",
    "(Gap-filled HABITAT-OSM)",
    "(Occupancy classification,\ntype-specific replacement costs)",
    "(Total floor area\nof residential stock)",
]
FINAL_LABEL = "Total\n(full uncertainty\nmodel median)"

SCENARIOS = ["SSP245", "SSP585"]
NA_COUNTRIES = ["USA", "CAN"]
RUS_COUNTRIES = ["RUS"]

# =========================
# Streletskiy et al. (2023) baselines (PPP USD 2024, billions)
# =========================
STRELETSKIY_BASELINE_B = {
    "SSP245": {"RUS": 63.0, "CAN": 2.4, "USA": 3.6},
    "SSP585": {"RUS": 96.0, "CAN": 2.8, "USA": 4.0},
}

# =========================
# Style
# =========================
C_BASELINE = "#9E9E9E"
C_TOTAL    = "#0B3C5D"
C_POS      = "#2CA02C"
C_NEG      = "#D62728"
C_LINE     = "#1F77B4"

# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_billions(x):
    return float(x) / 1e9

def _strip_cols(df):
    df.columns = df.columns.astype(str).str.strip()
    return df

def detect_cost_col(df, candidates):
    cols = [c.strip() for c in df.columns]
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Could not find any of {candidates} in columns: {cols[:40]}...")

def pooled_country_medians_from_mc(path, value_candidates, filter_mode_all=False):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = _strip_cols(pd.read_csv(path))

    cost_col = detect_cost_col(df, value_candidates)

    need = {"scenario", "shapeGroup", cost_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing required columns: {missing}")

    if filter_mode_all and "mode" in df.columns:
        df = df[df["mode"].astype(str).str.strip().str.upper() == "ALL"].copy()

    df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()
    df["shapeGroup"] = df["shapeGroup"].astype(str).str.strip().str.upper()
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

    out = (
        df.groupby(["scenario", "shapeGroup"], dropna=False)[cost_col]
          .median()  # FULL pooling over hazard_map × draw (and anything else)
          .reset_index()
          .rename(columns={cost_col: "value"})
    )
    out["value_b"] = out["value"].apply(to_billions)
    return out[["scenario", "shapeGroup", "value_b"]]

def fmt_delta(d):
    return f"{d:+.1f}"

def plot_waterfall(ax, baseline, block_vals, total_all, xticklabels, title, ylim_top):
    x = np.arange(6)
    deltas = [
        block_vals[0] - baseline,
        block_vals[1] - block_vals[0],
        block_vals[2] - block_vals[1],
        block_vals[3] - block_vals[2],
    ]

    OFFSET = ylim_top * 0.02

    ax.bar(x[0], baseline, width=0.75, color=C_BASELINE)
    ax.text(x[0], baseline + OFFSET, f"{baseline:.1f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

    prev = baseline
    cum_points = [baseline]
    for i, d in enumerate(deltas, start=1):
        new = prev + d
        bottom = min(prev, new)
        height = abs(d)
        color = C_POS if d >= 0 else C_NEG
        ax.bar(x[i], height, bottom=bottom, width=0.75, color=color)

        y_edge = bottom + height if d >= 0 else bottom
        y_label = y_edge + OFFSET if d >= 0 else y_edge - OFFSET

        ax.text(x[i], y_label, fmt_delta(d),
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=11,
                fontweight="bold")

        prev = new
        cum_points.append(prev)

    ax.bar(x[5], total_all, width=0.75, color=C_TOTAL)
    ax.text(x[5], total_all + OFFSET, f"{total_all:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    line_y = cum_points + [total_all]
    ax.plot(x, line_y, marker="o", linewidth=1.5, color=C_LINE)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=11, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, ylim_top)
    ax.grid(True, axis="y", alpha=0.3)

def build_tables():
    # B1–B4 from decomposition MC COUNTRY draws (PPP)
    block_tables = {}
    for b, fp in BLOCK_FILES.items():
        block_tables[b] = pooled_country_medians_from_mc(
            fp,
            value_candidates=["dam_cost_ppp2024", "dam_cost_usd2024", "dam_cost"],
            filter_mode_all=False
        ).rename(columns={"value_b": b})

    # merge B1–B4
    merged = block_tables["B1"]
    for b in ["B2", "B3", "B4"]:
        merged = merged.merge(block_tables[b], on=["scenario", "shapeGroup"], how="outer")

    # B5 from ALL uncertainty MC COUNTRY draws (USD2024 in your case)
    all_med = pooled_country_medians_from_mc(
        FULL_ALL_MC_COUNTRY,
        value_candidates=["dam_cost_usd2024", "dam_cost_ppp2024", "dam_cost"],
        filter_mode_all=True
    ).rename(columns={"value_b": "ALL_total_b"})

    merged = merged.merge(all_med, on=["scenario", "shapeGroup"], how="left")
    return merged

def determine_ylim(merged, countries):
    max_val = 0.0
    for scen in SCENARIOS:
        for ctry in countries:
            base = float(STRELETSKIY_BASELINE_B[scen][ctry])
            r = merged[(merged["scenario"] == scen) & (merged["shapeGroup"] == ctry)]
            if r.empty:
                max_val = max(max_val, base)
                continue

            vals = [base]
            for b in ["B1", "B2", "B3", "B4"]:
                v = r[b].iloc[0] if b in r.columns else np.nan
                if pd.notna(v):
                    vals.append(float(v))
            allv = r["ALL_total_b"].iloc[0] if "ALL_total_b" in r.columns else np.nan
            if pd.notna(allv):
                vals.append(float(allv))

            max_val = max(max_val, max(vals))
    return max(10.0, max_val * 1.25)

def plot_group(merged, countries, out_name, suptitle):
    ncols = len(countries)
    ylim_top = determine_ylim(merged, countries)

    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(6 * ncols, 8), sharey=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    xticklabels = ["Streletskiy (2023)"] + BLOCK_LABELS + [FINAL_LABEL]
    title_map = {"USA": "USA (Alaska)", "CAN": "Canada", "RUS": "Russia"}

    for i, scen in enumerate(SCENARIOS):
        for j, ctry in enumerate(countries):
            ax = axes[i, j]
            baseline = float(STRELETSKIY_BASELINE_B[scen][ctry])

            r = merged[(merged["scenario"] == scen) & (merged["shapeGroup"] == ctry)]
            if r.empty:
                block_vals = [baseline] * 4
                total_all = baseline
            else:
                # absolute totals after each block (billions)
                b1 = float(r["B1"].iloc[0]) if pd.notna(r["B1"].iloc[0]) else baseline
                b2 = float(r["B2"].iloc[0]) if pd.notna(r["B2"].iloc[0]) else b1
                b3 = float(r["B3"].iloc[0]) if pd.notna(r["B3"].iloc[0]) else b2
                b4 = float(r["B4"].iloc[0]) if pd.notna(r["B4"].iloc[0]) else b3
                block_vals = [b1, b2, b3, b4]

                allv = r["ALL_total_b"].iloc[0]
                total_all = float(allv) if (pd.notna(allv) and np.isfinite(allv)) else b4

            plot_waterfall(
                ax=ax,
                baseline=baseline,
                block_vals=block_vals,
                total_all=total_all,
                xticklabels=xticklabels,
                title=f"{scen} — {title_map.get(ctry, ctry)}",
                ylim_top=ylim_top
            )

            if i == 0:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if j == 0:
                ax.set_ylabel("Damaged replacement cost\n(PPP USD 2024, billions)")

    fig.suptitle(suptitle, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    ensure_dir(FIG_DIR)
    out_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_path, dpi=1000, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("[SUCCESS] Wrote:", out_path)

def main():
    ensure_dir(FIG_DIR)
    merged = build_tables()

    print("\n[EXTRACTED TOTALS (billions)]")
    print(merged.sort_values(["scenario", "shapeGroup"]).to_string(index=False))

    plot_group(
        merged=merged,
        countries=NA_COUNTRIES,
        out_name="waterfall_decomposition_NA_ppp2024.png",
        suptitle="Mid-century permafrost-related building damages — North America"
    )

    plot_group(
        merged=merged,
        countries=RUS_COUNTRIES,
        out_name="waterfall_decomposition_RUS_ppp2024.png",
        suptitle="Mid-century permafrost-related building damages — Russia"
    )

if __name__ == "__main__":
    main()
