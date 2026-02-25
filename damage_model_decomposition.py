# -*- coding: utf-8 -*-
"""
BC damage decomposition experiments (stepwise upgrades) + optional FS variability
using continuous BC loss from netCDF.

Cost modes:
- "combined_region": uses COMBINED_COST_PER_AREA (same for res/nonres)
- "median_country": uses MEDIAN_COST_PER_M2_NOMINAL (same for res/nonres)
- "occupancy_region": uses RES_COST_PER_AREA / NONRES_COST_PER_AREA (Value==1 is res)

All unit costs are treated as nominal and scaled to PPP USD 2024 with:
PPP_INFLATION_FACTOR_2024
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

OUT_DIR = r"D:\PhD_main\chapter_2\outputs\damage_decomposition_sampled_FS"
OUT_BASE = "BC_decomposition_v1"

SCENARIO_PATTERNS = {
    "SSP245": re.compile("ssp245", re.IGNORECASE),
    "SSP585": re.compile("ssp585", re.IGNORECASE),
}

ALLOWED_COUNTRIES = {"CAN", "RUS", "USA"}

# =========================================================
# COST SCALING
# =========================================================
PPP_INFLATION_FACTOR_2024 = 1.138574943035849

# =========================================================
# FS CONFIG
# =========================================================
RANDOM_SEED = 7

FS_RANGES = {
    "USA": (2.5, 3.0),
    "CAN": (2.5, 3.0),
    "RUS": (1.05, 1.56),
}

FIXED_FS_BASELINE = 1.15  # threshold computed as 1 - 1/fs

# =========================================================
# EXTENT MULTIPLIERS (deterministic in all modes here)
# =========================================================
EXTENT_BOUNDS = {
    "C": (0.90, 1.00),
    "D": (0.50, 0.90),
    "S": (0.10, 0.50),
    "I": (0.00, 0.00),
}

BASELINE_EXTENT = {
    "C": float(np.mean(EXTENT_BOUNDS["C"])),
    "D": float(np.mean(EXTENT_BOUNDS["D"])),
    "S": float(np.mean(EXTENT_BOUNDS["S"])),
    "I": 0.0,
}

# =========================================================
# netCDF time handling
# =========================================================
TIME_REDUCTION = "mean"  # "mean" or "first"

# =========================================================
# MODES (the 5 blocks)
# =========================================================
MODES_TO_RUN = [
    "B1_OSM_2D_COMBINEDCOST",
    "B2_OSM_HABITAT_2D_COMBINEDCOST",
    "B3_ADD_OCCUPANCY_COSTS",
    "B4_ADD_FLOORAREA",
    # "B5_FS_VARIABILITY",
]

MC_DRAWS = 300

MC_DRAWS_BY_MODE = {
    "B1_OSM_2D_COMBINEDCOST": MC_DRAWS,
    "B2_OSM_HABITAT_2D_COMBINEDCOST": MC_DRAWS,
    "B3_ADD_OCCUPANCY_COSTS": MC_DRAWS,
    "B4_ADD_FLOORAREA": MC_DRAWS,
    # "B5_FS_VARIABILITY": 300,
}

# Mode configuration switches
# cost_mode: "combined_region" | "median_country" | "occupancy_region"
MODE_CFG = {
    # Block 1 (UPDATED): OSM only, 2D, COMBINED_COST_PER_AREA by region
    "B1_OSM_2D_COMBINEDCOST": dict(
        sources={"OSM"},
        cost_mode="combined_region",
        use_floor_area=False,
        fs_mode="sample",
    ),
    # Block 2 (UPDATED): OSM+HABITAT, 2D, COMBINED_COST_PER_AREA by region
    "B2_OSM_HABITAT_2D_COMBINEDCOST": dict(
        sources={"OSM", "HABITAT"},
        cost_mode="combined_region",
        use_floor_area=False,
        fs_mode="sample",
    ),
    # Block 3: occupancy-specific costs by region, 2D
    "B3_ADD_OCCUPANCY_COSTS": dict(
        sources={"OSM", "HABITAT"},
        cost_mode="occupancy_region",
        use_floor_area=False,
        fs_mode="sample",
    ),
    # Block 4: occupancy-specific costs by region, floor area
    "B4_ADD_FLOORAREA": dict(
        sources={"OSM", "HABITAT"},
        cost_mode="occupancy_region",
        use_floor_area=True,
        fs_mode="sample",
    ),
}

# =========================================================
# OPTIONAL: WRITE PER-DRAW OUTPUTS
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


def threshold_from_fs(fs: float) -> float:
    fs = float(fs)
    return 1.0 - 1.0 / fs


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
    df = pd.read_csv(COST_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = {"shapeGroup", "shapeName", "RES_COST_PER_AREA", "NONRES_COST_PER_AREA", "COMBINED_COST_PER_AREA"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "COST_CSV is missing required columns: "
            + ", ".join(sorted(missing))
            + "\nExpected: shapeGroup, shapeName, RES_COST_PER_AREA, NONRES_COST_PER_AREA, COMBINED_COST_PER_AREA"
        )

    out = df[["shapeGroup", "shapeName", "RES_COST_PER_AREA", "NONRES_COST_PER_AREA", "COMBINED_COST_PER_AREA"]].copy()
    return out


def load_buildings(cost_df):
    bld = gpd.read_file(BUILDINGS_GPKG, layer=BUILDINGS_LAYER)
    bld = bld[bld["shapeGroup"].isin(ALLOWED_COUNTRIES)].copy()

    bld["ntl_group"] = np.where(
        bld["ntl_status"].astype(str).str.upper() == "ACTIVE",
        "ACTIVE",
        "NOT_ACTIVE",
    )

    # Merge cost inventory
    bld = bld.merge(cost_df, on=["shapeGroup", "shapeName"], how="left")

    # Occupancy flag
    bld["is_res"] = bld["Value"] == 1

    # Stories
    if "num_stories" in bld.columns:
        bld["num_stories"] = pd.to_numeric(bld["num_stories"], errors="coerce")
    else:
        bld["num_stories"] = np.nan

    if "EXTENT" not in bld.columns:
        raise KeyError("Buildings layer missing required 'EXTENT' column (C/D/S/I).")

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

    if vmax <= 0 and vmin < 0:
        vals = -vals

    finite2 = vals[np.isfinite(vals)]
    vmax2 = float(np.nanmax(finite2)) if finite2.size else 0.0
    if vmax2 > 1.5:
        vals = vals * 0.01

    loss = np.clip(np.where(np.isfinite(vals), vals, 0.0), 0.0, 1.0)

    ds.close()
    return loss


def build_unit_cost_per_m2(
    shapeGroup: np.ndarray,
    is_res: np.ndarray,
    cost_mode: str,
    res_cost_per_area: np.ndarray,
    nonres_cost_per_area: np.ndarray,
    combined_cost_per_area: np.ndarray,
) -> np.ndarray:
    """
    Returns unit cost per m^2 (nominal), before PPP scaling.

    cost_mode:
      - "combined_region": COMBINED_COST_PER_AREA (same for all buildings in region)
      - "median_country": MEDIAN_COST_PER_M2_NOMINAL (same for all buildings in country)
      - "occupancy_region": RES_COST_PER_AREA / NONRES_COST_PER_AREA by is_res
    """
    cost_mode = str(cost_mode).strip().lower()

    if cost_mode == "combined_region":
        unit_cost = combined_cost_per_area.astype(np.float64)
        if np.isnan(unit_cost).any():
            raise ValueError(
                "COMBINED_COST_PER_AREA contains NaNs after COST_CSV merge.\n"
                "Fix: ensure every (shapeGroup, shapeName) in buildings exists in ACPR cost inventory."
            )
        return unit_cost

    if cost_mode == "occupancy_region":
        unit_cost = np.where(is_res, res_cost_per_area, nonres_cost_per_area).astype(np.float64)
        if np.isnan(unit_cost).any():
            raise ValueError(
                "RES_COST_PER_AREA / NONRES_COST_PER_AREA contains NaNs after COST_CSV merge.\n"
                "Fix: ensure every (shapeGroup, shapeName) in buildings exists in ACPR cost inventory."
            )
        return unit_cost

    if cost_mode == "median_country":
        unit_cost = np.zeros_like(shapeGroup, dtype=np.float64)
        for c in ("USA", "CAN", "RUS"):
            unit_cost[shapeGroup == c] = float(MEDIAN_COST_PER_M2_NOMINAL[c])
        return unit_cost

    raise ValueError(f"Unknown cost_mode='{cost_mode}'. Expected: combined_region | median_country | occupancy_region")


def compute_effective_area(
    shape_area: np.ndarray,
    is_res: np.ndarray,
    num_stories: np.ndarray,
    use_floor_area: bool,
) -> np.ndarray:
    if not use_floor_area:
        return shape_area.astype(np.float64)

    out = shape_area.astype(np.float64).copy()
    has_stories = np.isfinite(num_stories)
    apply_story = is_res & has_stories
    out[apply_story] = shape_area[apply_story] * np.maximum(1.0, num_stories[apply_story])
    return out


def process_map_decomposition(
    nc_path,
    bld,
    scenario,
    mode,
    mc_draws,
    fs_thresholds,   # <-- NEW
    out_draw_country_csv,
    out_draw_region_csv,
):
    cfg = MODE_CFG[mode]
    D = int(mc_draws)

    # Filter sources per mode
    b = bld[bld["Source"].astype(str).isin(cfg["sources"])].copy()
    if b.empty:
        raise RuntimeError(f"MODE={mode} produced empty building set after Source filter: {cfg['sources']}")

    # centroids in EPSG:4326
    cent = b.geometry.centroid
    b_ll = b.copy()
    b_ll["geometry"] = cent
    if b_ll.crs is None:
        raise RuntimeError("Buildings CRS is missing.")
    b_ll = b_ll.to_crs("EPSG:4326")
    lon_pts = b_ll.geometry.x.to_numpy(np.float64)
    lat_pts = b_ll.geometry.y.to_numpy(np.float64)

    loss = read_bc_loss_from_netcdf(nc_path, lon_pts, lat_pts)

    # Grouping for outputs
    gkey = b[["shapeGroup", "shapeName", "Source", "ntl_group"]].astype(str).agg("|".join, axis=1)
    gcode, gunique = pd.factorize(gkey, sort=False)
    ng = len(gunique)
    gcode = gcode.astype(np.int32)

    # Region-only and country-only codes for per-draw outputs
    rkey = b[["shapeGroup", "shapeName"]].astype(str).agg("|".join, axis=1)
    rcode, runique = pd.factorize(rkey, sort=False)
    nr = len(runique)
    rcode = rcode.astype(np.int32)

    ckey = b["shapeGroup"].astype(str)
    ccode, cunique = pd.factorize(ckey, sort=False)
    nc = len(cunique)
    ccode = ccode.astype(np.int32)

    runique_split = pd.Series(runique).str.split("|", expand=True)
    runique_shapeGroup = runique_split[0].to_numpy(str)
    runique_shapeName = runique_split[1].to_numpy(str)
    cunique_shapeGroup = np.array(cunique, dtype=str)

    # Arrays
    N = len(b)
    shapeGroup = b["shapeGroup"].to_numpy(str)
    extent = b["EXTENT"].astype(str).str.upper().to_numpy(str)

    shape_area = b["Shape_Area"].to_numpy(np.float64)
    is_res = b["is_res"].to_numpy(bool)
    num_stories = b["num_stories"].to_numpy(np.float64)

    res_cost_per_area = b["RES_COST_PER_AREA"].to_numpy(np.float64)
    nonres_cost_per_area = b["NONRES_COST_PER_AREA"].to_numpy(np.float64)
    combined_cost_per_area = b["COMBINED_COST_PER_AREA"].to_numpy(np.float64)

    # EXTENT multipliers (deterministic)
    mult = np.zeros(N, dtype=np.float64)
    mult[extent == "C"] = float(BASELINE_EXTENT["C"])
    mult[extent == "D"] = float(BASELINE_EXTENT["D"])
    mult[extent == "S"] = float(BASELINE_EXTENT["S"])
    mult[extent == "I"] = 0.0

    # Output storages per draw
    tot_area = np.zeros((ng, D), dtype=np.float64)
    tot_cost = np.zeros((ng, D), dtype=np.float64)
    dam_area = np.zeros((ng, D), dtype=np.float64)
    dam_cost = np.zeros((ng, D), dtype=np.float64)

    tot_area_res = np.zeros((ng, D), dtype=np.float64)
    tot_area_non = np.zeros((ng, D), dtype=np.float64)
    tot_cost_res = np.zeros((ng, D), dtype=np.float64)
    tot_cost_non = np.zeros((ng, D), dtype=np.float64)

    dam_area_res = np.zeros((ng, D), dtype=np.float64)
    dam_area_non = np.zeros((ng, D), dtype=np.float64)
    dam_cost_res = np.zeros((ng, D), dtype=np.float64)
    dam_cost_non = np.zeros((ng, D), dtype=np.float64)

    # Per-draw region/country totals (damages only)
    reg_area = np.zeros((nr, D), dtype=np.float64)
    reg_cost = np.zeros((nr, D), dtype=np.float64)
    ctry_area = np.zeros((nc, D), dtype=np.float64)
    ctry_cost = np.zeros((nc, D), dtype=np.float64)

    hazard_name = os.path.basename(nc_path)

    region_draw_rows = []
    country_draw_rows = []
    stride = max(1, int(DRAW_STRIDE))

    for d in range(D):
        # Area basis
        area_eff = compute_effective_area(
            shape_area=shape_area,
            is_res=is_res,
            num_stories=num_stories,
            use_floor_area=bool(cfg["use_floor_area"]),
        )

        # Unit cost per m^2 (UPDATED: blocks 1–2 now use combined_region)
        unit_cost_nominal = build_unit_cost_per_m2(
            shapeGroup=shapeGroup,
            is_res=is_res,
            cost_mode=cfg["cost_mode"],
            res_cost_per_area=res_cost_per_area,
            nonres_cost_per_area=nonres_cost_per_area,
            combined_cost_per_area=combined_cost_per_area,
        )

        # Convert to PPP USD 2024
        unit_cost_ppp2024 = unit_cost_nominal * float(PPP_INFLATION_FACTOR_2024)

        # Total replacement cost
        cost_eff = area_eff * unit_cost_ppp2024

        # Thresholding
        if cfg["fs_mode"] == "fixed":
            fs = float(cfg["fixed_fs"])
            t = threshold_from_fs(fs)
            damaged = (loss > t).astype(np.float64)

        elif cfg["fs_mode"] == "sample":
            t = np.zeros(N, dtype=np.float64)
            t[shapeGroup == "USA"] = fs_thresholds["USA"][d]
            t[shapeGroup == "CAN"] = fs_thresholds["CAN"][d]
            t[shapeGroup == "RUS"] = fs_thresholds["RUS"][d]

            damaged = (loss > t).astype(np.float64)

        else:
            raise ValueError(f"Unknown fs_mode='{cfg['fs_mode']}' for MODE={mode}")

        # Totals
        ta = area_eff
        tc = cost_eff

        ta_res = np.where(is_res, ta, 0.0)
        ta_non = np.where(~is_res, ta, 0.0)
        tc_res = np.where(is_res, tc, 0.0)
        tc_non = np.where(~is_res, tc, 0.0)

        tot_area[:, d] = np.bincount(gcode, weights=ta, minlength=ng)
        tot_cost[:, d] = np.bincount(gcode, weights=tc, minlength=ng)
        tot_area_res[:, d] = np.bincount(gcode, weights=ta_res, minlength=ng)
        tot_area_non[:, d] = np.bincount(gcode, weights=ta_non, minlength=ng)
        tot_cost_res[:, d] = np.bincount(gcode, weights=tc_res, minlength=ng)
        tot_cost_non[:, d] = np.bincount(gcode, weights=tc_non, minlength=ng)

        # Damages
        da = area_eff * mult * damaged
        dc = cost_eff * mult * damaged

        da_res = np.where(is_res, da, 0.0)
        da_non = np.where(~is_res, da, 0.0)
        dc_res = np.where(is_res, dc, 0.0)
        dc_non = np.where(~is_res, dc, 0.0)

        dam_area[:, d] = np.bincount(gcode, weights=da, minlength=ng)
        dam_cost[:, d] = np.bincount(gcode, weights=dc, minlength=ng)
        dam_area_res[:, d] = np.bincount(gcode, weights=da_res, minlength=ng)
        dam_area_non[:, d] = np.bincount(gcode, weights=da_non, minlength=ng)
        dam_cost_res[:, d] = np.bincount(gcode, weights=dc_res, minlength=ng)
        dam_cost_non[:, d] = np.bincount(gcode, weights=dc_non, minlength=ng)

        # Per-draw region/country totals
        reg_area[:, d] = np.bincount(rcode, weights=da, minlength=nr)
        reg_cost[:, d] = np.bincount(rcode, weights=dc, minlength=nr)
        ctry_area[:, d] = np.bincount(ccode, weights=da, minlength=nc)
        ctry_cost[:, d] = np.bincount(ccode, weights=dc, minlength=nc)

        # Draw outputs (stride)
        if WRITE_MC_DRAWS and (d % stride == 0):
            if WRITE_REGION_DRAWS:
                region_draw_rows.append(pd.DataFrame({
                    "mode": mode,
                    "scenario": scenario,
                    "hazard_map": hazard_name,
                    "draw": int(d),
                    "shapeGroup": runique_shapeGroup,
                    "shapeName": runique_shapeName,
                    "dam_cost_ppp2024": reg_cost[:, d],
                    "dam_area_m2": reg_area[:, d],
                }))
            if WRITE_COUNTRY_DRAWS:
                country_draw_rows.append(pd.DataFrame({
                    "mode": mode,
                    "scenario": scenario,
                    "hazard_map": hazard_name,
                    "draw": int(d),
                    "shapeGroup": cunique_shapeGroup,
                    "dam_cost_ppp2024": ctry_cost[:, d],
                    "dam_area_m2": ctry_area[:, d],
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

    # Summaries
    s_ta = summarize_draws(tot_area)
    s_tc = summarize_draws(tot_cost)
    s_da = summarize_draws(dam_area)
    s_dc = summarize_draws(dam_cost)

    s_ta_r = summarize_draws(tot_area_res)
    s_ta_n = summarize_draws(tot_area_non)
    s_tc_r = summarize_draws(tot_cost_res)
    s_tc_n = summarize_draws(tot_cost_non)

    s_da_r = summarize_draws(dam_area_res)
    s_da_n = summarize_draws(dam_area_non)
    s_dc_r = summarize_draws(dam_cost_res)
    s_dc_n = summarize_draws(dam_cost_non)

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
        out[f"total_cost_ppp2024_{k}"] = s_tc[k]
        out[f"dam_area_m2_{k}"] = s_da[k]
        out[f"dam_cost_ppp2024_{k}"] = s_dc[k]

        out[f"total_area_res_m2_{k}"] = s_ta_r[k]
        out[f"total_area_nonres_m2_{k}"] = s_ta_n[k]
        out[f"total_cost_res_ppp2024_{k}"] = s_tc_r[k]
        out[f"total_cost_nonres_ppp2024_{k}"] = s_tc_n[k]

        out[f"dam_area_res_m2_{k}"] = s_da_r[k]
        out[f"dam_area_nonres_m2_{k}"] = s_da_n[k]
        out[f"dam_cost_res_ppp2024_{k}"] = s_dc_r[k]
        out[f"dam_cost_nonres_ppp2024_{k}"] = s_dc_n[k]

    out = out.drop(columns=["_gkey"])
    return out


def write_mode_summaries(per_map_mode, out_dir, out_base, mode):
    group_cols = ["mode", "scenario", "shapeGroup", "shapeName", "Source", "ntl_group"]
    numeric_cols = per_map_mode.select_dtypes(include=[np.number]).columns.tolist()

    for stat in ["mean", "min", "max"]:
        out = per_map_mode.groupby(group_cols, dropna=False)[numeric_cols].agg(stat).reset_index()

        out["prop_damaged_area"] = out["dam_area_m2_mean"] / out["total_area_m2_mean"]
        out["prop_damaged_cost"] = out["dam_cost_ppp2024_mean"] / out["total_cost_ppp2024_mean"]

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
            out["total_cost_res_ppp2024_mean"] > 0,
            out["dam_cost_res_ppp2024_mean"] / out["total_cost_res_ppp2024_mean"],
            np.nan,
        )
        out["prop_damaged_cost_nonres"] = np.where(
            out["total_cost_nonres_ppp2024_mean"] > 0,
            out["dam_cost_nonres_ppp2024_mean"] / out["total_cost_nonres_ppp2024_mean"],
            np.nan,
        )

        out_path = os.path.join(out_dir, f"{out_base}__{mode}_{stat}.csv")
        out.to_csv(out_path, index=False)

    print(f"[SUCCESS] Wrote {mode} summaries: {out_base}__{mode}_[mean|min|max].csv")

def generate_fs_threshold_draws(n_draws, seed):
    rng = np.random.default_rng(seed)

    fs_USA = rng.uniform(*FS_RANGES["USA"], size=n_draws)
    fs_CAN = rng.uniform(*FS_RANGES["CAN"], size=n_draws)
    fs_RUS = rng.uniform(*FS_RANGES["RUS"], size=n_draws)

    t_USA = 1.0 - 1.0 / fs_USA
    t_CAN = 1.0 - 1.0 / fs_CAN
    t_RUS = 1.0 - 1.0 / fs_RUS

    return {
        "USA": t_USA,
        "CAN": t_CAN,
        "RUS": t_RUS,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUT_DIR)

    fs_thresholds = generate_fs_threshold_draws(
        n_draws=MC_DRAWS,
        seed=RANDOM_SEED,
    )

    found = discover_hazard_nc()
    if not any(found.values()):
        raise RuntimeError(f"No netCDF hazard files found in: {HAZARD_NC_DIR}")

    cost_df = load_cost_inventory()
    bld = load_buildings(cost_df)

    for m in MODES_TO_RUN:
        if m not in MODE_CFG:
            raise KeyError(f"MODE '{m}' missing from MODE_CFG.")
        if m not in MC_DRAWS_BY_MODE:
            raise KeyError(f"MODE '{m}' missing from MC_DRAWS_BY_MODE.")

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
        cfg = MODE_CFG[mode]
        print(
            f"\n[MODE] {mode} | MC_DRAWS_PER_MAP={mc_draws} | "
            f"sources={sorted(cfg['sources'])} | cost_mode={cfg['cost_mode']} | "
            f"use_floor_area={cfg['use_floor_area']} | fs_mode={cfg['fs_mode']} | fixed_fs={cfg.get('fixed_fs', None)}"
        )

        for scenario, maps in found.items():
            if not maps:
                continue

            for nc in maps:
                df = process_map_decomposition(
                    nc_path=nc,
                    bld=bld,
                    scenario=scenario,
                    mode=mode,
                    mc_draws=mc_draws,
                    fs_thresholds=fs_thresholds,  # <-- NEW
                    out_draw_country_csv=out_country_draws,
                    out_draw_region_csv=out_region_draws,
                )
                if df is not None and not df.empty:
                    records.append(df)

        if not records:
            raise RuntimeError(f"No hazard maps produced any results for MODE={mode}")

        per_map_mode = pd.concat(records, ignore_index=True)
        write_mode_summaries(per_map_mode, OUT_DIR, OUT_BASE, mode)

        if WRITE_MC_DRAWS:
            if WRITE_COUNTRY_DRAWS:
                print(f"[SUCCESS] Wrote per-draw country totals: {out_country_draws}")
            if WRITE_REGION_DRAWS:
                print(f"[SUCCESS] Wrote per-draw region totals : {out_region_draws}")

    print("\n[SUCCESS] Completed all decomposition modes.")


if __name__ == "__main__":
    main()