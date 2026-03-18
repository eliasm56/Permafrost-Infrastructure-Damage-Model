# -*- coding: utf-8 -*-
"""
Bearing capacity (BC) building damage aggregation with proportional metrics + uncertainty propagation.

Base outputs (always written) for EACH uncertainty run-mode:
- <OUT_BASE>__<MODE>_mean.csv  (mean across hazard fields)
- <OUT_BASE>__<MODE>_min.csv   (min across hazard fields)
- <OUT_BASE>__<MODE>_max.csv   (max across hazard fields)

Optional extra outputs (if WRITE_MC_DRAWS=True) for EACH run-mode:
- <OUT_BASE>__<MODE>_MC_country_draws.csv   (per draw, per hazard field, country totals)
- <OUT_BASE>__<MODE>_MC_region_draws.csv    (per draw, per hazard field, region totals)

Split in the summary CSVs by:
scenario, country (Country), region (Region), Source

Uncertainty propagation (Monte Carlo) per hazard field includes (when enabled by MODE):
- FS threshold uncertainty (USA/CAN vs RUS)
- PF_zone multiplier uncertainty
- Optional hazard scaling (lognormal)
- Optional exposure scaling (lognormal)
- HABITAT detection F1 -> missing-stock multiplier
- Type label F1 -> probabilistic flips (res<->nonres)
- Story underestimation -> probabilistic +0/+1/+2 for residential w/ stories

This version directly uses the following BC change fields joined to the buildings layer:
SSP245:
- bc_diff_ssp245_AWI_CM_1_1_MR_2055_2064_2015_2024_nomask
- bc_diff_ssp245_MPI_ESM1_2_HR_2055_2064_2015_2024_nomask
- bc_diff_ssp245_NorESM2_MM_2055_2064_2015_2024_nomask
- bc_diff_ssp245_CESM2_WACCM_2055_2064_2015_2024

SSP585:
- bc_diff_ssp585_AWI_CM_1_1_MR_2055_2064_2015_2024_nomask
- bc_diff_ssp585_MPI_ESM1_2_HR_2055_2064_2015_2024_nomask
- bc_diff_ssp585_NorESM2_MM_2055_2064_2015_2024_nomask
- bc_diff_ssp585_CESM2_WACCM_2055_2064_2015_2024
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd

# =========================================================
# INPUTS
# =========================================================
BC_JOIN_CSV = "data/building_bc_change_by_hazard_map.csv"
BUILDINGS_GPKG = "data/HABITAT_OSM_bldgs_ADC.gpkg"
BUILDINGS_LAYER = "buildings"

COST_CSV = "data/ACPR_Adm1_cost_inventory.csv"

OUT_DIR = "outputs/damage_results"
OUT_BASE = "BC_risk_analysis_v3_uncert_fields"

GDP_DEFLATOR_2021_TO_2024 = 1.138574943035849

ALLOWED_COUNTRIES = {"CAN", "RUS", "USA"}

# BC hazard fields already stored in the buildings layer
HAZARD_FIELDS = {
    "SSP245": [
        "bc_diff_ssp245_AWI_CM_1_1_MR_2055_2064_2015_2024_nomask",
        "bc_diff_ssp245_MPI_ESM1_2_HR_2055_2064_2015_2024_nomask",
        "bc_diff_ssp245_NorESM2_MM_2055_2064_2015_2024_nomask",
        "bc_diff_ssp245_CESM2_WACCM_2055_2064_2015_2024",
    ],
    "SSP585": [
        "bc_diff_ssp585_AWI_CM_1_1_MR_2055_2064_2015_2024_nomask",
        "bc_diff_ssp585_MPI_ESM1_2_HR_2055_2064_2015_2024_nomask",
        "bc_diff_ssp585_NorESM2_MM_2055_2064_2015_2024_nomask",
        "bc_diff_ssp585_CESM2_WACCM_2055_2064_2015_2024",
    ],
}

# =========================================================
# UNCERTAINTY CONFIG (global defaults; per-mode overrides below)
# =========================================================
RANDOM_SEED = 7

# Safety factor ranges (by Country)
FS_RANGES = {
    "USA": (2.5, 3.0),
    "CAN": (2.5, 3.0),
    "RUS": (1.05, 1.56),
}

# PF_zone multiplier uncertainty bounds (by PF_zone category)
PF_zone_BOUNDS = {
    "C": (0.90, 1.00),
    "D": (0.50, 0.90),
    "S": (0.10, 0.50),
    "I": (0.00, 0.00),
}

# Optional structural/exposure uncertainty
HAZARD_SIGMA = 0.0
AREA_SIGMA = 0.0

# HABITAT + attributes uncertainty
HABITAT_DET_F1 = 0.79
TYPE_F1_RES = 0.84
TYPE_F1_NONRES = 0.82

# Story bias model
DELTA_STORY_VALUES = np.array([0, 1, 2], dtype=np.int32)
DELTA_STORY_PROBS = np.array([0.2, 0.6, 0.2], dtype=np.float64)

# =========================================================
# BASELINES (used when a source is "OFF" in a MODE)
# =========================================================
BASELINE_FS = {
    "USA": float(np.mean(FS_RANGES["USA"])),
    "CAN": float(np.mean(FS_RANGES["CAN"])),
    "RUS": float(np.mean(FS_RANGES["RUS"])),
}

BASELINE_PF_zone = {
    "C": float(np.mean(PF_zone_BOUNDS["C"])),
    "D": float(np.mean(PF_zone_BOUNDS["D"])),
    "S": float(np.mean(PF_zone_BOUNDS["S"])),
    "I": 0.0,
}

BASELINE_DET_MULT = 1.0
BASELINE_DELTA_STORY = 0.0
BASELINE_EPS_H = 1.0
BASELINE_EPS_A = 1.0

# =========================================================
# MODES TO RUN
# =========================================================
MODES_TO_RUN = [
    "ALL",
    "FS_ONLY",
    "PF_zone_ONLY",
    "DETECTION_ONLY",
    "TYPE_ONLY",
    "STORIES_ONLY",
    # "HAZARD_SCALE_ONLY",
    # "AREA_SCALE_ONLY",
]

MC_DRAWS_BY_MODE = {
    "ALL": 300,
    "FS_ONLY": 75,
    "PF_zone_ONLY": 75,
    "DETECTION_ONLY": 75,
    "TYPE_ONLY": 75,
    "STORIES_ONLY": 75,
    "HAZARD_SCALE_ONLY": 75,
    "AREA_SCALE_ONLY": 75,
}

MODE_FLAGS = {
    "ALL": dict(
        fs=True, PF_zone=True, hazard_scale=(HAZARD_SIGMA > 0), area_scale=(AREA_SIGMA > 0),
        detection=True, type_flips=True, stories=True
    ),
    "FS_ONLY": dict(fs=True, PF_zone=False, hazard_scale=False, area_scale=False, detection=False, type_flips=False, stories=False),
    "PF_zone_ONLY": dict(fs=False, PF_zone=True, hazard_scale=False, area_scale=False, detection=False, type_flips=False, stories=False),
    "DETECTION_ONLY": dict(fs=False, PF_zone=False, hazard_scale=False, area_scale=False, detection=True, type_flips=False, stories=False),
    "TYPE_ONLY": dict(fs=False, PF_zone=False, hazard_scale=False, area_scale=False, detection=False, type_flips=True, stories=False),
    "STORIES_ONLY": dict(fs=False, PF_zone=False, hazard_scale=False, area_scale=False, detection=False, type_flips=False, stories=True),
    "HAZARD_SCALE_ONLY": dict(fs=False, PF_zone=False, hazard_scale=True, area_scale=False, detection=False, type_flips=False, stories=False),
    "AREA_SCALE_ONLY": dict(fs=False, PF_zone=False, hazard_scale=False, area_scale=True, detection=False, type_flips=False, stories=False),
}

HAZARD_SIGMA_FOR_SCALE_ONLY = 0.15
AREA_SIGMA_FOR_SCALE_ONLY = 0.10

# =========================================================
# OPTIONAL: WRITE PER-DRAW OUTPUTS
# =========================================================
WRITE_MC_DRAWS = True
WRITE_COUNTRY_DRAWS = True
WRITE_REGION_DRAWS = True
DRAW_STRIDE = 1

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def threshold_from_fs(fs):
    fs = float(fs)
    return 1.0 - 1.0 / fs


def sample_lognormal_median1(rng, sigma):
    if sigma <= 0:
        return 1.0
    return float(rng.lognormal(mean=0.0, sigma=sigma))


def summarize_draws(arr_2d):
    return {
        "mean": arr_2d.mean(axis=1),
        "p05": np.percentile(arr_2d, 5, axis=1),
        "p50": np.percentile(arr_2d, 50, axis=1),
        "p95": np.percentile(arr_2d, 95, axis=1),
    }


def _append_csv(df, path):
    if df is None or df.empty:
        return
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def load_cost_inventory():
    print("[INFO] Loading replacement cost inventory...")
    df = pd.read_csv(COST_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df[["Country", "Region", "RES_COST_PER_AREA", "NONRES_COST_PER_AREA"]]


def discover_hazard_fields():
    found = {"SSP245": [], "SSP585": []}
    for scenario, fields in HAZARD_FIELDS.items():
        found[scenario] = list(fields)
    return found


def normalize_bc_loss(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=np.float64)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=np.float64)

    vmax = float(np.nanmax(finite))
    vmin = float(np.nanmin(finite))

    # If values are stored as negative decreases, flip sign
    if vmax <= 0 and vmin < 0:
        vals = -vals

    finite2 = vals[np.isfinite(vals)]
    vmax2 = float(np.nanmax(finite2)) if finite2.size else 0.0

    # If values look like percentages, convert to fractions
    if vmax2 > 1.5:
        vals = vals * 0.01

    loss = np.clip(np.where(np.isfinite(vals), vals, 0.0), 0.0, 1.0)
    return loss


def join_bc_fields_to_buildings(gpkg_path, gpkg_layer, csv_path, join_key="bldg_id"):
    """
    Read buildings from GeoPackage, join BC hazard-map fields from CSV on join_key,
    and return the joined GeoDataFrame for use inside the damage model script.

    Expected behavior
    -----------------
    - Keeps all GeoPackage building records
    - Adds all non-key CSV fields
    - Errors if join_key is missing
    - Errors if CSV has duplicate join keys
    - Warns if some buildings do not match a CSV record
    - Warns if some CSV records do not match a building
    """

    print("[INFO] Reading BC-change CSV...")
    bc_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    bc_df.columns = bc_df.columns.str.strip()
    print(f"[INFO] BC CSV rows: {len(bc_df):,}")

    print("[INFO] Reading buildings GeoPackage layer...")
    bld = gpd.read_file(gpkg_path, layer=gpkg_layer)
    bld.columns = bld.columns.str.strip()
    print(f"[INFO] Building rows: {len(bld):,}")

    if join_key not in bc_df.columns:
        raise KeyError(f"Join key '{join_key}' not found in BC CSV.")
    if join_key not in bld.columns:
        raise KeyError(f"Join key '{join_key}' not found in buildings layer.")

    # Standardize join key dtype
    bc_df[join_key] = bc_df[join_key].astype(str).str.strip()
    bld[join_key] = bld[join_key].astype(str).str.strip()

    # CSV must be one row per building
    dup_csv = bc_df[bc_df.duplicated(subset=[join_key], keep=False)].sort_values(join_key)
    if not dup_csv.empty:
        raise ValueError(
            f"BC CSV contains duplicate '{join_key}' values. "
            f"Example duplicates:\n{dup_csv[[join_key]].head(20)}"
        )

    # Avoid overwriting existing building columns unless they are the join key
    csv_nonkey_cols = [c for c in bc_df.columns if c != join_key]
    overlapping = [c for c in csv_nonkey_cols if c in bld.columns]
    if overlapping:
        print(f"[WARN] These CSV columns already exist in buildings and will overwrite: {overlapping}")
        bld = bld.drop(columns=overlapping)

    # Diagnostics before join
    bld_keys = set(bld[join_key])
    csv_keys = set(bc_df[join_key])

    n_bld_only = len(bld_keys - csv_keys)
    n_csv_only = len(csv_keys - bld_keys)
    n_match = len(bld_keys & csv_keys)

    print(f"[INFO] Matching {join_key} values: {n_match:,}")
    if n_bld_only > 0:
        print(f"[WARN] Buildings with no BC CSV match: {n_bld_only:,}")
    if n_csv_only > 0:
        print(f"[WARN] BC CSV rows with no building match: {n_csv_only:,}")

    print("[INFO] Joining BC fields to buildings...")
    bld = bld.merge(bc_df, on=join_key, how="left", validate="many_to_one")

    # Report null counts for joined hazard fields
    joined_cols = [c for c in bc_df.columns if c != join_key]
    if joined_cols:
        null_summary = bld[joined_cols].isna().all(axis=1).sum()
        print(f"[INFO] Buildings with all joined BC fields null: {null_summary:,}")

    print("[INFO] Join complete.")
    return bld


def load_buildings(cost_df):
    print("[INFO] Reading buildings layer and joining BC fields...")
    bld = join_bc_fields_to_buildings(
        gpkg_path=BUILDINGS_GPKG,
        gpkg_layer=BUILDINGS_LAYER,
        csv_path=BC_JOIN_CSV,
        join_key="bldg_id",
    )
    print(f"[INFO] Loaded {len(bld):,} total building records after join.")

    bld = bld[bld["Country"].isin(ALLOWED_COUNTRIES)].copy()
    print(f"[INFO] Retained {len(bld):,} buildings in allowed countries {sorted(ALLOWED_COUNTRIES)}.")

    required_cols = [
        "Country", "Region", "Source", "Occ_type", "Area", "PF_zone", "bldg_id"
    ]
    missing_required = [c for c in required_cols if c not in bld.columns]
    if missing_required:
        raise KeyError(f"Buildings layer missing required columns: {missing_required}")

    bld = bld.merge(cost_df, on=["Country", "Region"], how="left")
    bld["is_res"] = bld["Occ_type"] == 1

    if "num_stories" in bld.columns:
        bld["num_stories"] = pd.to_numeric(bld["num_stories"], errors="coerce")
    else:
        print("[WARN] 'num_stories' not found. Setting all stories to NaN.")
        bld["num_stories"] = np.nan

    if "PF_zone" not in bld.columns:
        raise KeyError("Buildings layer missing required 'PF_zone' column (C/D/S/I).")

    if bld["RES_COST_PER_AREA"].isna().any() or bld["NONRES_COST_PER_AREA"].isna().any():
        missing = bld.loc[
            bld["RES_COST_PER_AREA"].isna() | bld["NONRES_COST_PER_AREA"].isna(),
            ["Country", "Region"],
        ].drop_duplicates()
        raise ValueError(f"Missing replacement costs for regions:\n{missing}")

    all_hazard_fields = [f for lst in HAZARD_FIELDS.values() for f in lst]
    missing_bc = [f for f in all_hazard_fields if f not in bld.columns]
    if missing_bc:
        raise KeyError(f"Buildings layer missing BC hazard fields after CSV join:\n{missing_bc}")

    bld["_bid"] = np.arange(len(bld), dtype=np.int64)

    print("[INFO] Buildings layer passed validation.")
    return bld


def get_mode_sigmas(mode):
    flags = MODE_FLAGS[mode]
    hz = 0.0
    ar = 0.0

    if flags.get("hazard_scale", False):
        hz = float(HAZARD_SIGMA if mode == "ALL" else HAZARD_SIGMA_FOR_SCALE_ONLY)
    if flags.get("area_scale", False):
        ar = float(AREA_SIGMA if mode == "ALL" else AREA_SIGMA_FOR_SCALE_ONLY)

    return hz, ar


def process_hazard_field_uncertainty(
    hazard_field, bld, rng, scenario, mode, mc_draws, out_draw_country_csv, out_draw_region_csv
):
    """
    Same logic as the netCDF version, except BC loss comes directly from a field
    in the buildings GeoPackage instead of extracting from a raster/netCDF.
    """
    flags = MODE_FLAGS[mode]
    hz_sigma, ar_sigma = get_mode_sigmas(mode)

    print(f"    [FIELD] {hazard_field}")
    loss = normalize_bc_loss(bld[hazard_field].values)

    # Grouping codes: region x source
    gkey = bld[["Country", "Region", "Source"]].astype(str).agg("|".join, axis=1)
    gcode, gunique = pd.factorize(gkey, sort=False)
    ng = len(gunique)
    gcode = gcode.astype(np.int32)

    # Region-only and country-only codes for per-draw outputs
    rkey = bld[["Country", "Region"]].astype(str).agg("|".join, axis=1)
    rcode, runique = pd.factorize(rkey, sort=False)
    nr = len(runique)
    rcode = rcode.astype(np.int32)

    ckey = bld["Country"].astype(str)
    ccode, cunique = pd.factorize(ckey, sort=False)
    nc = len(cunique)
    ccode = ccode.astype(np.int32)

    runique_split = pd.Series(runique).str.split("|", expand=True)
    runique_Country = runique_split[0].to_numpy(str)
    runique_Region = runique_split[1].to_numpy(str)
    cunique_Country = np.array(cunique, dtype=str)

    N = len(bld)
    Country = bld["Country"].to_numpy(str)
    PF_zone = bld["PF_zone"].astype(str).str.upper().to_numpy(str)

    footprint_area = pd.to_numeric(bld["Area"], errors="coerce").to_numpy(np.float64)
    base_is_res = bld["is_res"].to_numpy(bool)
    num_stories = bld["num_stories"].to_numpy(np.float64)
    has_stories = np.isfinite(num_stories)

    res_cost_per_area = bld["RES_COST_PER_AREA"].to_numpy(np.float64)
    nonres_cost_per_area = bld["NONRES_COST_PER_AREA"].to_numpy(np.float64)

    mC_mask = (PF_zone == "C")
    mD_mask = (PF_zone == "D")
    mS_mask = (PF_zone == "S")
    mI_mask = (PF_zone == "I")

    D = int(mc_draws)

    # totals + damages (by gcode)
    tot_area = np.zeros((ng, D), dtype=np.float64)
    tot_cost = np.zeros((ng, D), dtype=np.float64)
    tot_area_res = np.zeros((ng, D), dtype=np.float64)
    tot_area_nonres = np.zeros((ng, D), dtype=np.float64)
    tot_cost_res = np.zeros((ng, D), dtype=np.float64)
    tot_cost_nonres = np.zeros((ng, D), dtype=np.float64)

    dam_area = np.zeros((ng, D), dtype=np.float64)
    dam_cost = np.zeros((ng, D), dtype=np.float64)
    dam_area_res = np.zeros((ng, D), dtype=np.float64)
    dam_area_nonres = np.zeros((ng, D), dtype=np.float64)
    dam_cost_res = np.zeros((ng, D), dtype=np.float64)
    dam_cost_nonres = np.zeros((ng, D), dtype=np.float64)

    # per-draw region/country totals
    reg_area = np.zeros((nr, D), dtype=np.float64)
    reg_cost = np.zeros((nr, D), dtype=np.float64)
    ctry_area = np.zeros((nc, D), dtype=np.float64)
    ctry_cost = np.zeros((nc, D), dtype=np.float64)

    reg_area_res = np.zeros((nr, D), dtype=np.float64)
    reg_area_nonres = np.zeros((nr, D), dtype=np.float64)
    reg_cost_res = np.zeros((nr, D), dtype=np.float64)
    reg_cost_nonres = np.zeros((nr, D), dtype=np.float64)

    ctry_area_res = np.zeros((nc, D), dtype=np.float64)
    ctry_area_nonres = np.zeros((nc, D), dtype=np.float64)
    ctry_cost_res = np.zeros((nc, D), dtype=np.float64)
    ctry_cost_nonres = np.zeros((nc, D), dtype=np.float64)

    # detection uncertainty triangular params
    det_mode = 1.0 - float(HABITAT_DET_F1)
    det_max = min(1.0 - 1e-9, 2.0 * det_mode)

    # Precompute baseline thresholds
    t_USA0 = threshold_from_fs(BASELINE_FS["USA"])
    t_CAN0 = threshold_from_fs(BASELINE_FS["CAN"])
    t_RUS0 = threshold_from_fs(BASELINE_FS["RUS"])

    # For per-draw outputs, write in batches
    region_draw_rows = []
    country_draw_rows = []

    stride = max(1, int(DRAW_STRIDE))
    progress_step = max(1, D // 10)

    for d in range(D):
        if (d == 0) or ((d + 1) % progress_step == 0) or (d == D - 1):
            print(f"      [DRAW] {d + 1}/{D}")

        # -------------------------------------------------
        # Detection uncertainty -> missing-stock multiplier
        # -------------------------------------------------
        if flags.get("detection", False):
            missed_frac = float(rng.triangular(0.0, det_mode, det_max))
            det_mult = 1.0 / (1.0 - missed_frac)
        else:
            det_mult = float(BASELINE_DET_MULT)

        # -------------------------------------------------
        # Type label uncertainty -> probabilistic flips
        # -------------------------------------------------
        is_res_draw = base_is_res.copy()
        if flags.get("type_flips", False):
            p_r2n = float(rng.uniform(0.0, 1.0 - float(TYPE_F1_RES)))
            p_n2r = float(rng.uniform(0.0, 1.0 - float(TYPE_F1_NONRES)))

            u = rng.random(N)
            res_mask = is_res_draw
            is_res_draw[res_mask & (u < p_r2n)] = False

            u2 = rng.random(N)
            non_mask = ~is_res_draw
            is_res_draw[non_mask & (u2 < p_n2r)] = True

        # -------------------------------------------------
        # Story underestimation correction (residential only)
        # -------------------------------------------------
        stories_draw = num_stories.copy()
        apply_story = is_res_draw & has_stories
        if flags.get("stories", False):
            delta_story = rng.choice(DELTA_STORY_VALUES, size=N, p=DELTA_STORY_PROBS).astype(np.float64)
            stories_draw[apply_story] = np.maximum(1.0, stories_draw[apply_story] + delta_story[apply_story])
        else:
            stories_draw[apply_story] = np.maximum(1.0, stories_draw[apply_story] + float(BASELINE_DELTA_STORY))

        floor_area_draw = footprint_area.copy()
        floor_area_draw[apply_story] = footprint_area[apply_story] * stories_draw[apply_story]

        # -------------------------------------------------
        # Optional exposure scaling (lognormal)
        # -------------------------------------------------
        if flags.get("area_scale", False):
            eps_a = sample_lognormal_median1(rng, ar_sigma)
        else:
            eps_a = float(BASELINE_EPS_A)

        area_eff = floor_area_draw * eps_a * det_mult

        unit_cost_2021 = np.where(is_res_draw, res_cost_per_area, nonres_cost_per_area)
        cost_eff = area_eff * unit_cost_2021 * GDP_DEFLATOR_2021_TO_2024

        # -------------------------------------------------
        # Optional hazard scaling + thresholding
        # -------------------------------------------------
        if flags.get("hazard_scale", False):
            eps_h = sample_lognormal_median1(rng, hz_sigma)
        else:
            eps_h = float(BASELINE_EPS_H)

        loss_eff = np.clip(loss * eps_h, 0.0, 1.0)

        # FS thresholds
        if flags.get("fs", False):
            t_USA = threshold_from_fs(rng.uniform(*FS_RANGES["USA"]))
            t_CAN = threshold_from_fs(rng.uniform(*FS_RANGES["CAN"]))
            t_RUS = threshold_from_fs(rng.uniform(*FS_RANGES["RUS"]))
        else:
            t_USA, t_CAN, t_RUS = t_USA0, t_CAN0, t_RUS0

        t = np.zeros(N, dtype=np.float64)
        t[Country == "USA"] = t_USA
        t[Country == "CAN"] = t_CAN
        t[Country == "RUS"] = t_RUS

        damaged = (loss_eff > t).astype(np.float64)

        # -------------------------------------------------
        # PF_zone multipliers uncertainty
        # -------------------------------------------------
        if flags.get("PF_zone", False):
            mC = rng.uniform(*PF_zone_BOUNDS["C"])
            mD = rng.uniform(*PF_zone_BOUNDS["D"])
            mS = rng.uniform(*PF_zone_BOUNDS["S"])
        else:
            mC = float(BASELINE_PF_zone["C"])
            mD = float(BASELINE_PF_zone["D"])
            mS = float(BASELINE_PF_zone["S"])

        mult = np.zeros(N, dtype=np.float64)
        mult[mC_mask] = mC
        mult[mD_mask] = mD
        mult[mS_mask] = mS
        mult[mI_mask] = 0.0

        # -------------------------------------------------
        # Totals
        # -------------------------------------------------
        ta = area_eff
        tc = cost_eff
        ta_res = np.where(is_res_draw, ta, 0.0)
        ta_non = np.where(~is_res_draw, ta, 0.0)
        tc_res = np.where(is_res_draw, tc, 0.0)
        tc_non = np.where(~is_res_draw, tc, 0.0)

        tot_area[:, d] = np.bincount(gcode, weights=ta, minlength=ng)
        tot_cost[:, d] = np.bincount(gcode, weights=tc, minlength=ng)
        tot_area_res[:, d] = np.bincount(gcode, weights=ta_res, minlength=ng)
        tot_area_nonres[:, d] = np.bincount(gcode, weights=ta_non, minlength=ng)
        tot_cost_res[:, d] = np.bincount(gcode, weights=tc_res, minlength=ng)
        tot_cost_nonres[:, d] = np.bincount(gcode, weights=tc_non, minlength=ng)

        # -------------------------------------------------
        # Damages
        # -------------------------------------------------
        da = area_eff * mult * damaged
        dc = cost_eff * mult * damaged

        da_res = np.where(is_res_draw, da, 0.0)
        da_non = np.where(~is_res_draw, da, 0.0)
        dc_res = np.where(is_res_draw, dc, 0.0)
        dc_non = np.where(~is_res_draw, dc, 0.0)

        dam_area[:, d] = np.bincount(gcode, weights=da, minlength=ng)
        dam_cost[:, d] = np.bincount(gcode, weights=dc, minlength=ng)
        dam_area_res[:, d] = np.bincount(gcode, weights=da_res, minlength=ng)
        dam_area_nonres[:, d] = np.bincount(gcode, weights=da_non, minlength=ng)
        dam_cost_res[:, d] = np.bincount(gcode, weights=dc_res, minlength=ng)
        dam_cost_nonres[:, d] = np.bincount(gcode, weights=dc_non, minlength=ng)

        # -------------------------------------------------
        # Per-draw country/region totals
        # -------------------------------------------------
        reg_area[:, d] = np.bincount(rcode, weights=da, minlength=nr)
        reg_cost[:, d] = np.bincount(rcode, weights=dc, minlength=nr)
        ctry_area[:, d] = np.bincount(ccode, weights=da, minlength=nc)
        ctry_cost[:, d] = np.bincount(ccode, weights=dc, minlength=nc)

        reg_area_res[:, d] = np.bincount(rcode, weights=da_res, minlength=nr)
        reg_area_nonres[:, d] = np.bincount(rcode, weights=da_non, minlength=nr)
        reg_cost_res[:, d] = np.bincount(rcode, weights=dc_res, minlength=nr)
        reg_cost_nonres[:, d] = np.bincount(rcode, weights=dc_non, minlength=nr)

        ctry_area_res[:, d] = np.bincount(ccode, weights=da_res, minlength=nc)
        ctry_area_nonres[:, d] = np.bincount(ccode, weights=da_non, minlength=nc)
        ctry_cost_res[:, d] = np.bincount(ccode, weights=dc_res, minlength=nc)
        ctry_cost_nonres[:, d] = np.bincount(ccode, weights=dc_non, minlength=nc)

        # -------------------------------------------------
        # Accumulate draw rows and flush periodically
        # -------------------------------------------------
        if WRITE_MC_DRAWS and (d % stride == 0):
            if WRITE_REGION_DRAWS:
                region_draw_rows.append(pd.DataFrame({
                    "mode": mode,
                    "scenario": scenario,
                    "hazard_map": hazard_field,
                    "draw": int(d),
                    "Country": runique_Country,
                    "Region": runique_Region,
                    "dam_cost_usd2024": reg_cost[:, d],
                    "dam_cost_res_usd2024": reg_cost_res[:, d],
                    "dam_cost_nonres_usd2024": reg_cost_nonres[:, d],
                    "dam_area_m2": reg_area[:, d],
                    "dam_area_res_m2": reg_area_res[:, d],
                    "dam_area_nonres_m2": reg_area_nonres[:, d],
                }))
            if WRITE_COUNTRY_DRAWS:
                country_draw_rows.append(pd.DataFrame({
                    "mode": mode,
                    "scenario": scenario,
                    "hazard_map": hazard_field,
                    "draw": int(d),
                    "Country": cunique_Country,
                    "dam_cost_usd2024": ctry_cost[:, d],
                    "dam_cost_res_usd2024": ctry_cost_res[:, d],
                    "dam_cost_nonres_usd2024": ctry_cost_nonres[:, d],
                    "dam_area_m2": ctry_area[:, d],
                    "dam_area_res_m2": ctry_area_res[:, d],
                    "dam_area_nonres_m2": ctry_area_nonres[:, d],
                }))

            if len(region_draw_rows) >= 25 and WRITE_REGION_DRAWS:
                _append_csv(pd.concat(region_draw_rows, ignore_index=True), out_draw_region_csv)
                region_draw_rows = []
            if len(country_draw_rows) >= 25 and WRITE_COUNTRY_DRAWS:
                _append_csv(pd.concat(country_draw_rows, ignore_index=True), out_draw_country_csv)
                country_draw_rows = []

    # final flush
    if WRITE_MC_DRAWS:
        if WRITE_REGION_DRAWS and region_draw_rows:
            _append_csv(pd.concat(region_draw_rows, ignore_index=True), out_draw_region_csv)
        if WRITE_COUNTRY_DRAWS and country_draw_rows:
            _append_csv(pd.concat(country_draw_rows, ignore_index=True), out_draw_country_csv)

    # ---------------------------------------------------------
    # Summaries (per grouping gcode)
    # ---------------------------------------------------------
    s_ta = summarize_draws(tot_area)
    s_tc = summarize_draws(tot_cost)
    s_ta_r = summarize_draws(tot_area_res)
    s_ta_n = summarize_draws(tot_area_nonres)
    s_tc_r = summarize_draws(tot_cost_res)
    s_tc_n = summarize_draws(tot_cost_nonres)

    s_da = summarize_draws(dam_area)
    s_dc = summarize_draws(dam_cost)
    s_da_r = summarize_draws(dam_area_res)
    s_da_n = summarize_draws(dam_area_nonres)
    s_dc_r = summarize_draws(dam_cost_res)
    s_dc_n = summarize_draws(dam_cost_nonres)

    out = pd.DataFrame({"_gkey": gunique})
    tmp = out["_gkey"].str.split("|", expand=True)
    out["Country"] = tmp[0]
    out["Region"] = tmp[1]
    out["Source"] = tmp[2]

    out["mode"] = mode
    out["hazard_map"] = hazard_field
    out["scenario"] = scenario

    for k in ("mean", "p05", "p50", "p95"):
        out[f"total_area_m2_{k}"] = s_ta[k]
        out[f"total_cost_usd2024_{k}"] = s_tc[k]
        out[f"total_area_res_m2_{k}"] = s_ta_r[k]
        out[f"total_area_nonres_m2_{k}"] = s_ta_n[k]
        out[f"total_cost_res_usd2024_{k}"] = s_tc_r[k]
        out[f"total_cost_nonres_usd2024_{k}"] = s_tc_n[k]

        out[f"dam_area_m2_{k}"] = s_da[k]
        out[f"dam_cost_usd2024_{k}"] = s_dc[k]
        out[f"dam_area_res_m2_{k}"] = s_da_r[k]
        out[f"dam_area_nonres_m2_{k}"] = s_da_n[k]
        out[f"dam_cost_res_usd2024_{k}"] = s_dc_r[k]
        out[f"dam_cost_nonres_usd2024_{k}"] = s_dc_n[k]

    out = out.drop(columns=["_gkey"])
    return out


def write_mode_summaries(per_map_mode, out_dir, out_base, mode):
    group_cols = ["mode", "scenario", "Country", "Region", "Source"]

    numeric_cols = per_map_mode.select_dtypes(include=[np.number]).columns.tolist()

    for stat in ["mean", "min", "max"]:
        out = per_map_mode.groupby(group_cols, dropna=False)[numeric_cols].agg(stat).reset_index()

        out["prop_damaged_area"] = out["dam_area_m2_mean"] / out["total_area_m2_mean"]
        out["prop_damaged_cost"] = out["dam_cost_usd2024_mean"] / out["total_cost_usd2024_mean"]

        out["prop_damaged_area_res"] = np.where(
            out["total_area_res_m2_mean"] > 0,
            out["dam_area_res_m2_mean"] / out["total_area_res_m2_mean"],
            np.nan,
        )
        out["prop_damaged_area_nonres"] = np.where(
            out["total_area_nonres_m2_mean"] > 0,
            out["dam_area_nonres_m2_mean"] / out["total_area_nonres_m2_mean"],
            np.nan,
        )
        out["prop_damaged_cost_res"] = np.where(
            out["total_cost_res_usd2024_mean"] > 0,
            out["dam_cost_res_usd2024_mean"] / out["total_cost_res_usd2024_mean"],
            np.nan,
        )
        out["prop_damaged_cost_nonres"] = np.where(
            out["total_cost_nonres_usd2024_mean"] > 0,
            out["dam_cost_nonres_usd2024_mean"] / out["total_cost_nonres_usd2024_mean"],
            np.nan,
        )

        out_path = os.path.join(out_dir, f"{out_base}__{mode}_{stat}.csv")
        out.to_csv(out_path, index=False)

    print(f"[SUCCESS] Wrote {mode} summaries: {out_base}__{mode}_[mean|min|max].csv")


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    rng = np.random.default_rng(RANDOM_SEED)

    cost_df = load_cost_inventory()
    bld = load_buildings(cost_df)

    found = discover_hazard_fields()
    if not any(found.values()):
        raise RuntimeError("No BC hazard fields configured.")

    for m in MODES_TO_RUN:
        if m not in MODE_FLAGS:
            raise KeyError(f"MODE '{m}' not found in MODE_FLAGS.")
        if m not in MC_DRAWS_BY_MODE:
            raise KeyError(f"MODE '{m}' not found in MC_DRAWS_BY_MODE.")

    total_fields = sum(len(v) for v in found.values())

    for mode in MODES_TO_RUN:
        mc_draws = int(MC_DRAWS_BY_MODE[mode])

        out_country_draws = os.path.join(OUT_DIR, f"{OUT_BASE}__{mode}_MC_country_draws.csv")
        out_region_draws = os.path.join(OUT_DIR, f"{OUT_BASE}__{mode}_MC_region_draws.csv")

        if WRITE_MC_DRAWS:
            if WRITE_COUNTRY_DRAWS and os.path.exists(out_country_draws):
                os.remove(out_country_draws)
            if WRITE_REGION_DRAWS and os.path.exists(out_region_draws):
                os.remove(out_region_draws)

        records = []

        print(f"\n[MODE] {mode} | MC_DRAWS_PER_MAP={mc_draws} | flags={MODE_FLAGS[mode]} | sigmas={get_mode_sigmas(mode)}")

        field_counter = 0
        for scenario, fields in found.items():
            if not fields:
                continue

            print(f"  [SCENARIO] {scenario} | {len(fields)} hazard fields")
            for hazard_field in fields:
                field_counter += 1
                print(f"  [PROGRESS] Hazard field {field_counter}/{total_fields}")

                df = process_hazard_field_uncertainty(
                    hazard_field=hazard_field,
                    bld=bld,
                    rng=rng,
                    scenario=scenario,
                    mode=mode,
                    mc_draws=mc_draws,
                    out_draw_country_csv=out_country_draws,
                    out_draw_region_csv=out_region_draws,
                )
                if df is not None and not df.empty:
                    records.append(df)

        if not records:
            raise RuntimeError(f"No hazard fields produced any results for MODE={mode}")

        per_map_mode = pd.concat(records, ignore_index=True)

        # Write mode summaries (mean/min/max across hazard fields)
        write_mode_summaries(per_map_mode, OUT_DIR, OUT_BASE, mode)

        if WRITE_MC_DRAWS:
            if WRITE_COUNTRY_DRAWS:
                print(f"[SUCCESS] Wrote per-draw country totals: {out_country_draws}")
            if WRITE_REGION_DRAWS:
                print(f"[SUCCESS] Wrote per-draw region totals : {out_region_draws}")

    print("\n[SUCCESS] Completed all modes.")


if __name__ == "__main__":
    main()
