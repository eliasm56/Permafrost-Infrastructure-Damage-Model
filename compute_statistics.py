# -*- coding: utf-8 -*-
"""
Compute P05 / median / P95 damage summaries from Monte Carlo draw CSVs.

Works for either:
- MC_country_draws.csv
- MC_region_draws.csv

Produces:
- Circumpolar totals
- Macro totals: North America (USA+CAN) vs Russia
"""

import os
import numpy as np
import pandas as pd

# =========================================================
# INPUT
# =========================================================
INPUT_CSV = r"C:\Users\manos\Desktop\research\BC_risk_analysis_v3_uncert_fields__ALL_MC_region_draws.csv"

OUT_DIR = r"C:\Users\manos\Desktop\research"
OUT_CIRCUMPOLAR_CSV = os.path.join(OUT_DIR, "summary_circumpolar.csv")
OUT_MACRO_CSV = os.path.join(OUT_DIR, "summary_macro.csv")

SCENARIOS = ["SSP245", "SSP585"]

# If None, script auto-detects:
#   dam_cost_usd2024 or dam_cost_ppp2024
DRAW_METRIC = None

# =========================================================
# HELPERS
# =========================================================
def detect_metric_column(df):
    candidates = ["dam_cost_usd2024", "dam_cost_ppp2024"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "Could not find a damage metric column. Expected one of: "
        + ", ".join(candidates)
    )


def detect_country_column(df):
    candidates = ["Country", "shapeGroup"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "Could not find a country column. Expected one of: "
        + ", ".join(candidates)
    )


def uncertainty_width_p05_p95(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    p05 = float(np.percentile(vals, 5))
    p50 = float(np.percentile(vals, 50))
    p95 = float(np.percentile(vals, 95))
    width = p95 - p05
    return p05, p50, p95, width


def add_macro_region(df, country_col):
    df = df.copy()
    df[country_col] = df[country_col].astype(str).str.upper().str.strip()

    df["macro"] = np.where(
        df[country_col].isin(["USA", "CAN"]),
        "North America",
        np.where(df[country_col] == "RUS", "Russia", "Other")
    )
    return df


def pooled_circumpolar_totals_by_draw(df, draw_metric):
    out = (
        df.groupby(["scenario", "hazard_map", "draw"], as_index=False)[draw_metric]
          .sum()
    )
    return out


def pooled_macro_totals_by_draw(df, draw_metric, country_col):
    df2 = add_macro_region(df, country_col=country_col)

    out = (
        df2.groupby(["scenario", "macro", "hazard_map", "draw"], as_index=False)[draw_metric]
           .sum()
    )
    return out


def compute_circumpolar_summary(df, draw_metric):
    tot = pooled_circumpolar_totals_by_draw(df, draw_metric=draw_metric)

    rows = []
    for scen in SCENARIOS:
        sub = tot[tot["scenario"] == scen]
        if sub.empty:
            continue

        vals = sub[draw_metric].to_numpy(np.float64)
        p05, p50, p95, width = uncertainty_width_p05_p95(vals)

        rows.append({
            "scenario": scen,
            "geography": "Circumpolar",
            "metric": draw_metric,
            "n_draw_totals": len(vals),
            "p05": p05,
            "p50": p50,
            "p95": p95,
            "width_p95_p05": width,
        })

    return pd.DataFrame(rows)


def compute_macro_summary(df, draw_metric, country_col):
    tot = pooled_macro_totals_by_draw(df, draw_metric=draw_metric, country_col=country_col)

    rows = []
    for scen in SCENARIOS:
        for macro in ["North America", "Russia"]:
            sub = tot[(tot["scenario"] == scen) & (tot["macro"] == macro)]
            if sub.empty:
                continue

            vals = sub[draw_metric].to_numpy(np.float64)
            p05, p50, p95, width = uncertainty_width_p05_p95(vals)

            rows.append({
                "scenario": scen,
                "geography": macro,
                "metric": draw_metric,
                "n_draw_totals": len(vals),
                "p05": p05,
                "p50": p50,
                "p95": p95,
                "width_p95_p05": width,
            })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    print("[INFO] Reading input CSV...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"[INFO] Rows read: {len(df):,}")

    required_base = {"scenario", "hazard_map", "draw"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise KeyError(f"Input CSV missing required columns: {sorted(missing_base)}")

    draw_metric = DRAW_METRIC if DRAW_METRIC is not None else detect_metric_column(df)
    country_col = detect_country_column(df)

    print(f"[INFO] Using damage metric column: {draw_metric}")
    print(f"[INFO] Using country column: {country_col}")

    df[draw_metric] = pd.to_numeric(df[draw_metric], errors="coerce")
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce").astype("Int64")

    circ = compute_circumpolar_summary(df, draw_metric=draw_metric)
    macro = compute_macro_summary(df, draw_metric=draw_metric, country_col=country_col)

    print("\n[CIRCUMPOLAR SUMMARY]")
    print(circ.to_string(index=False))

    print("\n[MACRO SUMMARY]")
    print(macro.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    circ.to_csv(OUT_CIRCUMPOLAR_CSV, index=False)
    macro.to_csv(OUT_MACRO_CSV, index=False)

    print(f"\n[SUCCESS] Wrote: {OUT_CIRCUMPOLAR_CSV}")
    print(f"[SUCCESS] Wrote: {OUT_MACRO_CSV}")


if __name__ == "__main__":
    main()
