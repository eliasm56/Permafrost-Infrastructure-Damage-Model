# -*- coding: utf-8 -*-
"""
BC damage aggregation with proportional metrics + uncertainty propagation
using continuous BC loss from netCDF.

Base outputs (always written) for EACH uncertainty run-mode:
- <OUT_BASE>__<MODE>_mean.csv  (mean across hazard maps)
- <OUT_BASE>__<MODE>_min.csv   (min across hazard maps)
- <OUT_BASE>__<MODE>_max.csv   (max across hazard maps)

Optional extra outputs (if WRITE_MC_DRAWS=True) for EACH run-mode:
- <OUT_BASE>__<MODE>_MC_country_draws.csv   (per draw, per hazard map, country totals)
- <OUT_BASE>__<MODE>_MC_region_draws.csv    (per draw, per hazard map, region totals)

Split in the summary CSVs by:
scenario, country (shapeGroup), region (shapeName), Source, NTL activity

Uncertainty propagation (Monte Carlo) per hazard map includes (when enabled by MODE):
- FS threshold uncertainty (USA/CAN vs RUS)
- EXTENT multiplier uncertainty
- Optional hazard scaling (lognormal)
- Optional exposure scaling (lognormal)
- HABITAT detection F1 -> missing-stock multiplier
- Type label F1 -> probabilistic flips (res<->nonres)
- Story underestimation -> probabilistic +0/+1/+2 for residential w/ stories

----------------------------------------
This script can run multiple "modes". Example set:
- ALL (full uncertainty)
- FS_ONLY
- EXTENT_ONLY
- DETECTION_ONLY
- TYPE_ONLY
- STORIES_ONLY
- HAZARD_SCALE_ONLY
- AREA_SCALE_ONLY

For modes that do NOT include a given uncertainty source, that source is held fixed at a
deterministic baseline (see BASELINE_* configs).

"""

import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

# =========================================================
# INPUTS
# =========================================================
HAZARD_NC_DIR = r"D:\Streletskiy hazard maps\netCDF\BC"

BC_VAR = "bc_change"
LON_NAME = "lon"
LAT_NAME = "lat"

BUILDINGS_GPKG = r"D:\PhD_main\chapter_2\outputs\HABITAT_OSM_bldg_type_activity_topo_with_stories.gpkg"
BUILDINGS_LAYER = "buildings"

COST_CSV = r"D:\PhD_main\chapter_3\data\costs\ACPR_Adm1_cost_inventory.csv"

OUT_DIR = r"D:\PhD_main\chapter_2\outputs\damage_model_w_uncertainty_run2"
OUT_BASE = "BC_risk_analysis_v3_uncert_netcdf"

GDP_DEFLATOR_2021_TO_2024 = 1.138574943035849

SCENARIO_PATTERNS = {
    "SSP245": re.compile("ssp245", re.IGNORECASE),
    "SSP585": re.compile("ssp585", re.IGNORECASE),
}

ALLOWED_COUNTRIES = {"CAN", "RUS", "USA"}

# =========================================================
# UNCERTAINTY CONFIG (global defaults; per-mode overrides below)
# =========================================================
RANDOM_SEED = 7

# Safety factor ranges (by shapeGroup)
FS_RANGES = {
    "USA": (2.5, 3.0),
    "CAN": (2.5, 3.0),
    "RUS": (1.05, 1.56),
}

# Extent multiplier uncertainty bounds (by EXTENT category)
EXTENT_BOUNDS = {
    "C": (0.90, 1.00),
    "D": (0.50, 0.90),
    "S": (0.10, 0.50),
    "I": (0.00, 0.00),
}

# Optional structural/exposure uncertainty (set to 0.0 to disable in ALL)
HAZARD_SIGMA = 0.0   # e.g., 0.15
AREA_SIGMA = 0.0     # e.g., 0.10

# HABITAT + attributes uncertainty
HABITAT_DET_F1 = 0.79
TYPE_F1_RES = 0.84
TYPE_F1_NONRES = 0.82

# Story bias model: underest by ~1 story
DELTA_STORY_VALUES = np.array([0, 1, 2], dtype=np.int32)
DELTA_STORY_PROBS = np.array([0.2, 0.6, 0.2], dtype=np.float64)

# netCDF time handling
TIME_REDUCTION = "mean"  # "mean" or "first"

# =========================================================
# BASELINES (used when a source is "OFF" in a MODE)
# =========================================================
# FS baseline uses midpoint FS -> threshold
BASELINE_FS = {
    "USA": float(np.mean(FS_RANGES["USA"])),
    "CAN": float(np.mean(FS_RANGES["CAN"])),
    "RUS": float(np.mean(FS_RANGES["RUS"])),
}

# EXTENT baseline uses midpoint multipliers
BASELINE_EXTENT = {
    "C": float(np.mean(EXTENT_BOUNDS["C"])),
    "D": float(np.mean(EXTENT_BOUNDS["D"])),
    "S": float(np.mean(EXTENT_BOUNDS["S"])),
    "I": 0.0,
}

# Detection baseline: no missing stock
BASELINE_DET_MULT = 1.0

# Type baseline: no flips
BASELINE_P_R2N = 0.0
BASELINE_P_N2R = 0.0

# Stories baseline: no correction (delta=0)
BASELINE_DELTA_STORY = 0.0

# Hazard/exposure scaling baseline
BASELINE_EPS_H = 1.0
BASELINE_EPS_A = 1.0

# =========================================================
# MODES TO RUN (OAT sensitivity)
# =========================================================
MODES_TO_RUN = [
    "ALL",
    "FS_ONLY",
    "EXTENT_ONLY",
    "DETECTION_ONLY",
    "TYPE_ONLY",
    "STORIES_ONLY",
    # "HAZARD_SCALE_ONLY",
    # "AREA_SCALE_ONLY",
]

MC_DRAWS_BY_MODE = {
    "ALL": 300,
    "FS_ONLY": 75,
    "EXTENT_ONLY": 75,
    "DETECTION_ONLY": 75,
    "TYPE_ONLY": 75,
    "STORIES_ONLY": 75,
    "HAZARD_SCALE_ONLY": 75,
    "AREA_SCALE_ONLY": 75,
}

MODE_FLAGS = {
    "ALL": dict(
        fs=True, extent=True, hazard_scale=(HAZARD_SIGMA > 0), area_scale=(AREA_SIGMA > 0),
        detection=True, type_flips=True, stories=True
    ),
    "FS_ONLY": dict(fs=True, extent=False, hazard_scale=False, area_scale=False, detection=False, type_flips=False, stories=False),
    "EXTENT_ONLY": dict(fs=False, extent=True, hazard_scale=False, area_scale=False, detection=False, type_flips=False, stories=False),
    "DETECTION_ONLY": dict(fs=False, extent=False, hazard_scale=False, area_scale=False, detection=True, type_flips=False, stories=False),
    "TYPE_ONLY": dict(fs=False, extent=False, hazard_scale=False, area_scale=False, detection=False, type_flips=True, stories=False),
    "STORIES_ONLY": dict(fs=False, extent=False, hazard_scale=False, area_scale=False, detection=False, type_flips=False, stories=True),
    "HAZARD_SCALE_ONLY": dict(fs=False, extent=False, hazard_scale=True, area_scale=False, detection=False, type_flips=False, stories=False),
    "AREA_SCALE_ONLY": dict(fs=False, extent=False, hazard_scale=False, area_scale=True, detection=False, type_flips=False, stories=False),
}

HAZARD_SIGMA_FOR_SCALE_ONLY = 0.15
AREA_SIGMA_FOR_SCALE_ONLY = 0.10

# =========================================================
# OPTIONAL: WRITE PER-DRAW OUTPUTS (FOR VIOLINS/DENSITIES)
# =========================================================
WRITE_MC_DRAWS = True
WRITE_COUNTRY_DRAWS = True
WRITE_REGION_DRAWS = True
DRAW_STRIDE = 1  # 1 keeps all draws; 2 keeps every other draw, etc.

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def discover_hazard_nc():
    found = {"SSP245": [], "SSP585": []}
    for nc in glob.glob(os.path.join(HAZARD_NC_DIR, "*.nc")):
        for scen, pat in SCENARIO_PATTERNS.items():
            if pat.search(nc):
                found[scen].append(nc)
    for k in found:
        found[k] = sorted(found[k])
    return found


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
    """Append df to CSV; write header if file doesn't exist."""
    if df is None or df.empty:
        return
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def load_cost_inventory():
    df = pd.read_csv(COST_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df[["shapeGroup", "shapeName", "RES_COST_PER_AREA", "NONRES_COST_PER_AREA"]]


def load_buildings(cost_df):
    bld = gpd.read_file(BUILDINGS_GPKG, layer=BUILDINGS_LAYER)
    bld = bld[bld["shapeGroup"].isin(ALLOWED_COUNTRIES)].copy()

    bld["ntl_group"] = np.where(
        bld["ntl_status"].astype(str).str.upper() == "ACTIVE",
        "ACTIVE",
        "NOT_ACTIVE",
    )

    bld = bld.merge(cost_df, on=["shapeGroup", "shapeName"], how="left")

    bld["is_res"] = bld["Value"] == 1

    if "num_stories" in bld.columns:
        bld["num_stories"] = pd.to_numeric(bld["num_stories"], errors="coerce")
    else:
        bld["num_stories"] = np.nan

    if "EXTENT" not in bld.columns:
        raise KeyError("Buildings layer missing required 'EXTENT' column (C/D/S/I).")

    if bld["RES_COST_PER_AREA"].isna().any() or bld["NONRES_COST_PER_AREA"].isna().any():
        missing = bld.loc[
            bld["RES_COST_PER_AREA"].isna() | bld["NONRES_COST_PER_AREA"].isna(),
            ["shapeGroup", "shapeName"],
        ].drop_duplicates()
        raise ValueError(f"Missing replacement costs for regions:\n{missing}")

    bld["_bid"] = np.arange(len(bld), dtype=np.int64)
    return bld


def _nearest_index_1d(sorted_vals, query_vals):
    idx = np.searchsorted(sorted_vals, query_vals, side="left")
    idx = np.clip(idx, 0, len(sorted_vals) - 1)
    prev = np.clip(idx - 1, 0, len(sorted_vals) - 1)
    d1 = np.abs(sorted_vals[idx] - query_vals)
    d0 = np.abs(sorted_vals[prev] - query_vals)
    use_prev = d0 <= d1
    idx[use_prev] = prev[use_prev]
    return idx


def read_bc_loss_from_netcdf(nc_path, lon_pts, lat_pts):
    ds = xr.open_dataset(nc_path)

    if BC_VAR not in ds:
        raise KeyError(f"{BC_VAR} not found in {nc_path}. Vars: {list(ds.data_vars)}")
    if LON_NAME not in ds.coords and LON_NAME not in ds:
        raise KeyError(f"{LON_NAME} not found in {nc_path}. Coords: {list(ds.coords)}")
    if LAT_NAME not in ds.coords and LAT_NAME not in ds:
        raise KeyError(f"{LAT_NAME} not found in {nc_path}. Coords: {list(ds.coords)}")

    da = ds[BC_VAR]

    if "time" in da.dims:
        if TIME_REDUCTION == "mean":
            da = da.mean(dim="time", skipna=True)
        else:
            da = da.isel(time=0)

    lon = ds[LON_NAME].values
    lat = ds[LAT_NAME].values

    if lon.ndim != 1 or lat.ndim != 1:
        raise RuntimeError(
            f"Expected 1D lon/lat grids but got lon.ndim={lon.ndim}, lat.ndim={lat.ndim}."
        )

    lon_asc = np.all(np.diff(lon) >= 0)
    lat_asc = np.all(np.diff(lat) >= 0)

    lon_vals = lon if lon_asc else lon[::-1]
    lat_vals = lat if lat_asc else lat[::-1]

    ix = _nearest_index_1d(lon_vals, lon_pts)
    iy = _nearest_index_1d(lat_vals, lat_pts)

    if not lon_asc:
        ix = (len(lon) - 1) - ix
    if not lat_asc:
        iy = (len(lat) - 1) - iy

    if (LAT_NAME not in da.dims) or (LON_NAME not in da.dims):
        raise RuntimeError(f"{BC_VAR} dims {da.dims} do not include {LAT_NAME} and {LON_NAME}.")

    da2 = da.transpose(LAT_NAME, LON_NAME)
    arr = da2.values
    vals = arr[iy, ix].astype(np.float64)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        ds.close()
        return np.zeros_like(vals, dtype=np.float64)

    vmax = float(np.nanmax(finite))
    vmin = float(np.nanmin(finite))

    # If mostly negative, interpret as decreases represented as negative
    if vmax <= 0 and vmin < 0:
        vals = -vals

    # If percent-like, scale to fraction
    finite2 = vals[np.isfinite(vals)]
    vmax2 = float(np.nanmax(finite2)) if finite2.size else 0.0
    if vmax2 > 1.5:
        vals = vals * 0.01

    loss = np.clip(np.where(np.isfinite(vals), vals, 0.0), 0.0, 1.0)

    ds.close()
    return loss


def get_mode_sigmas(mode):
    """Return (hazard_sigma, area_sigma) to use for a mode."""
    flags = MODE_FLAGS[mode]
    hz = 0.0
    ar = 0.0

    if flags.get("hazard_scale", False):
        if mode == "ALL":
            hz = float(HAZARD_SIGMA)
        else:
            hz = float(HAZARD_SIGMA_FOR_SCALE_ONLY)
    if flags.get("area_scale", False):
        if mode == "ALL":
            ar = float(AREA_SIGMA)
        else:
            ar = float(AREA_SIGMA_FOR_SCALE_ONLY)

    return hz, ar


def process_map_uncertainty_netcdf(
    nc_path, bld, rng, scenario, mode, mc_draws, out_draw_country_csv, out_draw_region_csv
):
    """
    Returns per-map summarized MC outputs for grouping:
      [shapeGroup, shapeName, Source, ntl_group]
    Optionally appends per-draw region/country totals for the hazard map.

    mode controls which uncertainty components are active.

    IMPORTANT:
    Per-draw outputs now include occupancy-split damages:
      - dam_cost_res_usd2024, dam_cost_nonres_usd2024
      - dam_area_res_m2, dam_area_nonres_m2
    """
    flags = MODE_FLAGS[mode]
    hz_sigma, ar_sigma = get_mode_sigmas(mode)

    # centroids in EPSG:4326
    cent = bld.geometry.centroid
    bld_ll = bld.copy()
    bld_ll["geometry"] = cent
    if bld_ll.crs is None:
        raise RuntimeError("Buildings CRS is missing.")
    bld_ll = bld_ll.to_crs("EPSG:4326")

    lon_pts = bld_ll.geometry.x.to_numpy(np.float64)
    lat_pts = bld_ll.geometry.y.to_numpy(np.float64)

    loss = read_bc_loss_from_netcdf(nc_path, lon_pts, lat_pts)  # (N,)

    # Grouping codes: region x source x activity (summary outputs)
    gkey = bld[["shapeGroup", "shapeName", "Source", "ntl_group"]].astype(str).agg("|".join, axis=1)
    gcode, gunique = pd.factorize(gkey, sort=False)
    ng = len(gunique)
    gcode = gcode.astype(np.int32)

    # Region-only and country-only codes for per-draw outputs
    rkey = bld[["shapeGroup", "shapeName"]].astype(str).agg("|".join, axis=1)
    rcode, runique = pd.factorize(rkey, sort=False)
    nr = len(runique)
    rcode = rcode.astype(np.int32)

    ckey = bld["shapeGroup"].astype(str)
    ccode, cunique = pd.factorize(ckey, sort=False)
    nc = len(cunique)
    ccode = ccode.astype(np.int32)

    runique_split = pd.Series(runique).str.split("|", expand=True)
    runique_shapeGroup = runique_split[0].to_numpy(str)
    runique_shapeName  = runique_split[1].to_numpy(str)
    cunique_shapeGroup = np.array(cunique, dtype=str)

    N = len(bld)
    shapeGroup = bld["shapeGroup"].to_numpy(str)
    extent = bld["EXTENT"].astype(str).str.upper().to_numpy(str)

    footprint_area = bld["Shape_Area"].to_numpy(np.float64)
    base_is_res = bld["is_res"].to_numpy(bool)
    num_stories = bld["num_stories"].to_numpy(np.float64)
    has_stories = np.isfinite(num_stories)

    res_cost_per_area = bld["RES_COST_PER_AREA"].to_numpy(np.float64)
    nonres_cost_per_area = bld["NONRES_COST_PER_AREA"].to_numpy(np.float64)

    mC_mask = (extent == "C")
    mD_mask = (extent == "D")
    mS_mask = (extent == "S")
    mI_mask = (extent == "I")

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

    # per-draw region/country totals (for optional draw outputs) - TOTAL
    reg_area = np.zeros((nr, D), dtype=np.float64)
    reg_cost = np.zeros((nr, D), dtype=np.float64)
    ctry_area = np.zeros((nc, D), dtype=np.float64)
    ctry_cost = np.zeros((nc, D), dtype=np.float64)

    # per-draw region/country totals - RES / NONRES  (FIX)
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

    # Precompute baseline thresholds (when FS is off)
    t_USA0 = threshold_from_fs(BASELINE_FS["USA"])
    t_CAN0 = threshold_from_fs(BASELINE_FS["CAN"])
    t_RUS0 = threshold_from_fs(BASELINE_FS["RUS"])

    hazard_name = os.path.basename(nc_path)

    # For per-draw outputs, we write in batches to avoid huge memory spikes
    region_draw_rows = []
    country_draw_rows = []

    stride = max(1, int(DRAW_STRIDE))

    for d in range(D):
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
        else:
            pass

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

        # floor area per draw
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
        t[shapeGroup == "USA"] = t_USA
        t[shapeGroup == "CAN"] = t_CAN
        t[shapeGroup == "RUS"] = t_RUS

        damaged = (loss_eff > t).astype(np.float64)

        # -------------------------------------------------
        # EXTENT multipliers uncertainty
        # -------------------------------------------------
        if flags.get("extent", False):
            mC = rng.uniform(*EXTENT_BOUNDS["C"])
            mD = rng.uniform(*EXTENT_BOUNDS["D"])
            mS = rng.uniform(*EXTENT_BOUNDS["S"])
        else:
            mC = float(BASELINE_EXTENT["C"])
            mD = float(BASELINE_EXTENT["D"])
            mS = float(BASELINE_EXTENT["S"])

        mult = np.zeros(N, dtype=np.float64)
        mult[mC_mask] = mC
        mult[mD_mask] = mD
        mult[mS_mask] = mS
        mult[mI_mask] = 0.0

        # -------------------------------------------------
        # Totals (denominators)
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
        # Damages (numerators)
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
        # Per-draw country/region totals for draw outputs
        # (collapse across Source + ntl_group)
        # -------------------------------------------------
        reg_area[:, d] = np.bincount(rcode, weights=da, minlength=nr)
        reg_cost[:, d] = np.bincount(rcode, weights=dc, minlength=nr)
        ctry_area[:, d] = np.bincount(ccode, weights=da, minlength=nc)
        ctry_cost[:, d] = np.bincount(ccode, weights=dc, minlength=nc)

        # FIX: occupancy-split per-draw region/country totals
        reg_area_res[:, d] = np.bincount(rcode, weights=da_res, minlength=nr)
        reg_area_nonres[:, d] = np.bincount(rcode, weights=da_non, minlength=nr)
        reg_cost_res[:, d] = np.bincount(rcode, weights=dc_res, minlength=nr)
        reg_cost_nonres[:, d] = np.bincount(rcode, weights=dc_non, minlength=nr)

        ctry_area_res[:, d] = np.bincount(ccode, weights=da_res, minlength=nc)
        ctry_area_nonres[:, d] = np.bincount(ccode, weights=da_non, minlength=nc)
        ctry_cost_res[:, d] = np.bincount(ccode, weights=dc_res, minlength=nc)
        ctry_cost_nonres[:, d] = np.bincount(ccode, weights=dc_non, minlength=nc)

        # -------------------------------------------------
        # Accumulate draw rows (stride) and flush periodically
        # -------------------------------------------------
        if WRITE_MC_DRAWS and (d % stride == 0):
            if WRITE_REGION_DRAWS:
                region_draw_rows.append(pd.DataFrame({
                    "mode": mode,
                    "scenario": scenario,
                    "hazard_map": hazard_name,
                    "draw": int(d),
                    "shapeGroup": runique_shapeGroup,
                    "shapeName": runique_shapeName,
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
                    "hazard_map": hazard_name,
                    "draw": int(d),
                    "shapeGroup": cunique_shapeGroup,
                    "dam_cost_usd2024": ctry_cost[:, d],
                    "dam_cost_res_usd2024": ctry_cost_res[:, d],
                    "dam_cost_nonres_usd2024": ctry_cost_nonres[:, d],
                    "dam_area_m2": ctry_area[:, d],
                    "dam_area_res_m2": ctry_area_res[:, d],
                    "dam_area_nonres_m2": ctry_area_nonres[:, d],
                }))

            # flush every ~25 kept draws to avoid big RAM
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
    out["shapeGroup"] = tmp[0]
    out["shapeName"] = tmp[1]
    out["Source"] = tmp[2]
    out["ntl_group"] = tmp[3]

    out["mode"] = mode
    out["hazard_map"] = hazard_name
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
    group_cols = ["mode", "scenario", "shapeGroup", "shapeName", "Source", "ntl_group"]

    # aggregate ONLY numeric columns
    numeric_cols = per_map_mode.select_dtypes(include=[np.number]).columns.tolist()

    for stat in ["mean", "min", "max"]:
        out = per_map_mode.groupby(group_cols, dropna=False)[numeric_cols].agg(stat).reset_index()

        # proportions using MC-mean totals and damages (consistent)
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

    found = discover_hazard_nc()
    if not any(found.values()):
        raise RuntimeError(f"No netCDF hazard files found in: {HAZARD_NC_DIR}")

    # Sanity: ensure all requested modes exist in configs
    for m in MODES_TO_RUN:
        if m not in MODE_FLAGS:
            raise KeyError(f"MODE '{m}' not found in MODE_FLAGS.")
        if m not in MC_DRAWS_BY_MODE:
            raise KeyError(f"MODE '{m}' not found in MC_DRAWS_BY_MODE.")

    for mode in MODES_TO_RUN:
        mc_draws = int(MC_DRAWS_BY_MODE[mode])

        # Draw output filenames per mode
        out_country_draws = os.path.join(OUT_DIR, f"{OUT_BASE}__{mode}_MC_country_draws.csv")
        out_region_draws  = os.path.join(OUT_DIR, f"{OUT_BASE}__{mode}_MC_region_draws.csv")

        # If writing MC draws, start fresh for mode
        if WRITE_MC_DRAWS:
            if WRITE_COUNTRY_DRAWS and os.path.exists(out_country_draws):
                os.remove(out_country_draws)
            if WRITE_REGION_DRAWS and os.path.exists(out_region_draws):
                os.remove(out_region_draws)

        records = []

        print(f"\n[MODE] {mode} | MC_DRAWS_PER_MAP={mc_draws} | flags={MODE_FLAGS[mode]} | sigmas={get_mode_sigmas(mode)}")

        for scenario, maps in found.items():
            if not maps:
                continue

            for nc in maps:
                df = process_map_uncertainty_netcdf(
                    nc_path=nc,
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
            raise RuntimeError(f"No hazard maps produced any results for MODE={mode}")

        per_map_mode = pd.concat(records, ignore_index=True)

        # Write mode summaries (mean/min/max across hazard maps)
        write_mode_summaries(per_map_mode, OUT_DIR, OUT_BASE, mode)

        # Report draw outputs
        if WRITE_MC_DRAWS:
            if WRITE_COUNTRY_DRAWS:
                print(f"[SUCCESS] Wrote per-draw country totals: {out_country_draws}")
            if WRITE_REGION_DRAWS:
                print(f"[SUCCESS] Wrote per-draw region totals : {out_region_draws}")

    print("\n[SUCCESS] Completed all modes.")


if __name__ == "__main__":
    main()
