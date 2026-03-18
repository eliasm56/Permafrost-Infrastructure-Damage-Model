# -*- coding: utf-8 -*-
"""
BC damage decomposition experiments (stepwise upgrades).

Cost modes:
- "combined_region": uses COMBINED_COST_PER_AREA (same for res/nonres)
- "occupancy_region": uses RES_COST_PER_AREA / NONRES_COST_PER_AREA (Occ_type==1 is res)

All unit costs are treated as nominal and scaled to USD 2024 with:
PPP_INFLATION_FACTOR_2024
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

OUT_DIR = "outputs"
OUT_BASE = "BC_decomposition_v1_fields"

ALLOWED_COUNTRIES = {"CAN", "RUS", "USA"}

# BC hazard fields already stored in the joined building table
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
# PF_zone MULTIPLIERS (deterministic in all modes here)
# =========================================================
PF_zone_BOUNDS = {
    "C": (0.90, 1.00),
    "D": (0.50, 0.90),
    "S": (0.10, 0.50),
    "I": (0.00, 0.00),
}

BASELINE_PF_zone = {
    "C": float(np.mean(PF_zone_BOUNDS["C"])),
    "D": float(np.mean(PF_zone_BOUNDS["D"])),
    "S": float(np.mean(PF_zone_BOUNDS["S"])),
    "I": 0.0,
}

# =========================================================
# MODES (the 4 blocks)
# =========================================================
MODES_TO_RUN = [
    "B1_OSM_2D_COMBINEDCOST",
    "B2_OSM_HABITAT_2D_COMBINEDCOST",
    "B3_ADD_OCCUPANCY_COSTS",
    "B4_ADD_FLOORAREA",
]

MC_DRAWS = 300

MC_DRAWS_BY_MODE = {
    "B1_OSM_2D_COMBINEDCOST": MC_DRAWS,
    "B2_OSM_HABITAT_2D_COMBINEDCOST": MC_DRAWS,
    "B3_ADD_OCCUPANCY_COSTS": MC_DRAWS,
    "B4_ADD_FLOORAREA": MC_DRAWS,
}

MODE_CFG = {
    "B1_OSM_2D_COMBINEDCOST": dict(
        sources={"OSM"},
        cost_mode="combined_region",
        use_floor_area=False,
        fs_mode="sample",
    ),
    "B2_OSM_HABITAT_2D_COMBINEDCOST": dict(
        sources={"OSM", "HABITAT"},
        cost_mode="combined_region",
        use_floor_area=False,
        fs_mode="sample",
    ),
    "B3_ADD_OCCUPANCY_COSTS": dict(
        sources={"OSM", "HABITAT"},
        cost_mode="occupancy_region",
        use_floor_area=False,
        fs_mode="sample",
    ),
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
DRAW_STRIDE = 1

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def discover_hazard_fields():
    found = {"SSP245": [], "SSP585": []}
    for scenario, fields in HAZARD_FIELDS.items():
        found[scenario] = list(fields)
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

    return np.clip(np.where(np.isfinite(vals), vals, 0.0), 0.0, 1.0)


def join_bc_fields_to_buildings(gpkg_path, gpkg_layer, csv_path, join_key="bldg_id"):
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

    bc_df[join_key] = bc_df[join_key].astype(str).str.strip()
    bld[join_key] = bld[join_key].astype(str).str.strip()

    dup_csv = bc_df[bc_df.duplicated(subset=[join_key], keep=False)].sort_values(join_key)
    if not dup_csv.empty:
        raise ValueError(
            f"BC CSV contains duplicate '{join_key}' values. "
            f"Example duplicates:\n{dup_csv[[join_key]].head(20)}"
        )

    csv_nonkey_cols = [c for c in bc_df.columns if c != join_key]
    overlapping = [c for c in csv_nonkey_cols if c in bld.columns]
    if overlapping:
        print(f"[WARN] These CSV columns already exist in buildings and will overwrite: {overlapping}")
        bld = bld.drop(columns=overlapping)

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

    joined_cols = [c for c in bc_df.columns if c != join_key]
    if joined_cols:
        null_summary = bld[joined_cols].isna().all(axis=1).sum()
        print(f"[INFO] Buildings with all joined BC fields null: {null_summary:,}")

    print("[INFO] Join complete.")
    return bld


def load_cost_inventory():
    print("[INFO] Loading replacement cost inventory...")
    df = pd.read_csv(COST_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = {
        "Country", "Region",
        "RES_COST_PER_AREA", "NONRES_COST_PER_AREA", "COMBINED_COST_PER_AREA"
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "COST_CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df[[
        "Country", "Region",
        "RES_COST_PER_AREA", "NONRES_COST_PER_AREA", "COMBINED_COST_PER_AREA"
    ]].copy()


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
    bld["is_res"] = pd.to_numeric(bld["Occ_type"], errors="coerce") == 1

    if "num_stories" in bld.columns:
        bld["num_stories"] = pd.to_numeric(bld["num_stories"], errors="coerce")
    else:
        print("[WARN] 'num_stories' not found. Setting all stories to NaN.")
        bld["num_stories"] = np.nan

    if "PF_zone" not in bld.columns:
        raise KeyError("Buildings layer missing required 'PF_zone' column (C/D/S/I).")

    if bld["RES_COST_PER_AREA"].isna().any() or bld["NONRES_COST_PER_AREA"].isna().any() or bld["COMBINED_COST_PER_AREA"].isna().any():
        missing = bld.loc[
            bld["RES_COST_PER_AREA"].isna()
            | bld["NONRES_COST_PER_AREA"].isna()
            | bld["COMBINED_COST_PER_AREA"].isna(),
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


def build_unit_cost_per_m2(
    is_res: np.ndarray,
    cost_mode: str,
    res_cost_per_area: np.ndarray,
    nonres_cost_per_area: np.ndarray,
    combined_cost_per_area: np.ndarray,
) -> np.ndarray:
    cost_mode = str(cost_mode).strip().lower()

    if cost_mode == "combined_region":
        unit_cost = combined_cost_per_area.astype(np.float64)
        if np.isnan(unit_cost).any():
            raise ValueError("COMBINED_COST_PER_AREA contains NaNs after COST_CSV merge.")
        return unit_cost

    if cost_mode == "occupancy_region":
        unit_cost = np.where(is_res, res_cost_per_area, nonres_cost_per_area).astype(np.float64)
        if np.isnan(unit_cost).any():
            raise ValueError("RES_COST_PER_AREA / NONRES_COST_PER_AREA contains NaNs after COST_CSV merge.")
        return unit_cost

    raise ValueError(f"Unknown cost_mode='{cost_mode}'. Expected: combined_region | occupancy_region")


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


def process_field_decomposition(
    hazard_field,
    bld,
    scenario,
    mode,
    mc_draws,
    fs_thresholds,
    out_draw_country_csv,
    out_draw_region_csv,
):
    cfg = MODE_CFG[mode]
    D = int(mc_draws)

    b = bld[bld["Source"].astype(str).isin(cfg["sources"])].copy()
    if b.empty:
        raise RuntimeError(f"MODE={mode} produced empty building set after Source filter: {cfg['sources']}")

    print(f"    [FIELD] {hazard_field}")
    loss = normalize_bc_loss(b[hazard_field].values)

    # Grouping for outputs
    gkey = b[["Country", "Region", "Source"]].astype(str).agg("|".join, axis=1)
    gcode, gunique = pd.factorize(gkey, sort=False)
    ng = len(gunique)
    gcode = gcode.astype(np.int32)

    rkey = b[["Country", "Region"]].astype(str).agg("|".join, axis=1)
    rcode, runique = pd.factorize(rkey, sort=False)
    nr = len(runique)
    rcode = rcode.astype(np.int32)

    ckey = b["Country"].astype(str)
    ccode, cunique = pd.factorize(ckey, sort=False)
    nc = len(cunique)
    ccode = ccode.astype(np.int32)

    runique_split = pd.Series(runique).str.split("|", expand=True)
    runique_Country = runique_split[0].to_numpy(str)
    runique_Region = runique_split[1].to_numpy(str)
    cunique_Country = np.array(cunique, dtype=str)

    N = len(b)
    Country = b["Country"].to_numpy(str)
    PF_zone = b["PF_zone"].astype(str).str.upper().to_numpy(str)

    shape_area = pd.to_numeric(b["Area"], errors="coerce").to_numpy(np.float64)
    is_res = b["is_res"].to_numpy(bool)
    num_stories = b["num_stories"].to_numpy(np.float64)

    res_cost_per_area = b["RES_COST_PER_AREA"].to_numpy(np.float64)
    nonres_cost_per_area = b["NONRES_COST_PER_AREA"].to_numpy(np.float64)
    combined_cost_per_area = b["COMBINED_COST_PER_AREA"].to_numpy(np.float64)

    # Deterministic PF_zone multipliers
    mult = np.zeros(N, dtype=np.float64)
    mult[PF_zone == "C"] = float(BASELINE_PF_zone["C"])
    mult[PF_zone == "D"] = float(BASELINE_PF_zone["D"])
    mult[PF_zone == "S"] = float(BASELINE_PF_zone["S"])
    mult[PF_zone == "I"] = 0.0

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

    reg_area = np.zeros((nr, D), dtype=np.float64)
    reg_cost = np.zeros((nr, D), dtype=np.float64)
    ctry_area = np.zeros((nc, D), dtype=np.float64)
    ctry_cost = np.zeros((nc, D), dtype=np.float64)

    region_draw_rows = []
    country_draw_rows = []
    stride = max(1, int(DRAW_STRIDE))
    progress_step = max(1, D // 10)

    for d in range(D):
        if (d == 0) or ((d + 1) % progress_step == 0) or (d == D - 1):
            print(f"      [DRAW] {d + 1}/{D}")

        area_eff = compute_effective_area(
            shape_area=shape_area,
            is_res=is_res,
            num_stories=num_stories,
            use_floor_area=bool(cfg["use_floor_area"]),
        )

        unit_cost_nominal = build_unit_cost_per_m2(
            is_res=is_res,
            cost_mode=cfg["cost_mode"],
            res_cost_per_area=res_cost_per_area,
            nonres_cost_per_area=nonres_cost_per_area,
            combined_cost_per_area=combined_cost_per_area,
        )

        unit_cost_2024 = unit_cost_nominal * float(PPP_INFLATION_FACTOR_2024)
        cost_eff = area_eff * unit_cost_2024

        if cfg["fs_mode"] == "fixed":
            fs = float(FIXED_FS_BASELINE)
            t = threshold_from_fs(fs)
            damaged = (loss > t).astype(np.float64)
        elif cfg["fs_mode"] == "sample":
            t = np.zeros(N, dtype=np.float64)
            t[Country == "USA"] = fs_thresholds["USA"][d]
            t[Country == "CAN"] = fs_thresholds["CAN"][d]
            t[Country == "RUS"] = fs_thresholds["RUS"][d]
            damaged = (loss > t).astype(np.float64)
        else:
            raise ValueError(f"Unknown fs_mode='{cfg['fs_mode']}' for MODE={mode}")

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

        reg_area[:, d] = np.bincount(rcode, weights=da, minlength=nr)
        reg_cost[:, d] = np.bincount(rcode, weights=dc, minlength=nr)
        ctry_area[:, d] = np.bincount(ccode, weights=da, minlength=nc)
        ctry_cost[:, d] = np.bincount(ccode, weights=dc, minlength=nc)

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
                    "dam_area_m2": reg_area[:, d],
                }))
            if WRITE_COUNTRY_DRAWS:
                country_draw_rows.append(pd.DataFrame({
                    "mode": mode,
                    "scenario": scenario,
                    "hazard_map": hazard_field,
                    "draw": int(d),
                    "Country": cunique_Country,
                    "dam_cost_usd2024": ctry_cost[:, d],
                    "dam_area_m2": ctry_area[:, d],
                }))

            if len(region_draw_rows) >= 25 and WRITE_REGION_DRAWS:
                _append_csv(pd.concat(region_draw_rows, ignore_index=True), out_draw_region_csv)
                region_draw_rows = []
            if len(country_draw_rows) >= 25 and WRITE_COUNTRY_DRAWS:
                _append_csv(pd.concat(country_draw_rows, ignore_index=True), out_draw_country_csv)
                country_draw_rows = []

    if WRITE_MC_DRAWS:
        if WRITE_REGION_DRAWS and region_draw_rows:
            _append_csv(pd.concat(region_draw_rows, ignore_index=True), out_draw_region_csv)
        if WRITE_COUNTRY_DRAWS and country_draw_rows:
            _append_csv(pd.concat(country_draw_rows, ignore_index=True), out_draw_country_csv)

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
    out["Country"] = tmp[0]
    out["Region"] = tmp[1]
    out["Source"] = tmp[2]

    out["mode"] = mode
    out["hazard_map"] = hazard_field
    out["scenario"] = scenario

    for k in ("mean", "p05", "p50", "p95"):
        out[f"total_area_m2_{k}"] = s_ta[k]
        out[f"total_cost_usd2024_{k}"] = s_tc[k]
        out[f"dam_area_m2_{k}"] = s_da[k]
        out[f"dam_cost_usd2024_{k}"] = s_dc[k]

        out[f"total_area_res_m2_{k}"] = s_ta_r[k]
        out[f"total_area_nonres_m2_{k}"] = s_ta_n[k]
        out[f"total_cost_res_usd2024_{k}"] = s_tc_r[k]
        out[f"total_cost_nonres_usd2024_{k}"] = s_tc_n[k]

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

    fs_thresholds = generate_fs_threshold_draws(
        n_draws=MC_DRAWS,
        seed=RANDOM_SEED,
    )

    found = discover_hazard_fields()
    if not any(found.values()):
        raise RuntimeError("No BC hazard fields configured.")

    cost_df = load_cost_inventory()
    bld = load_buildings(cost_df)

    for m in MODES_TO_RUN:
        if m not in MODE_CFG:
            raise KeyError(f"MODE '{m}' missing from MODE_CFG.")
        if m not in MC_DRAWS_BY_MODE:
            raise KeyError(f"MODE '{m}' missing from MC_DRAWS_BY_MODE.")

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
        cfg = MODE_CFG[mode]
        print(
            f"\n[MODE] {mode} | MC_DRAWS_PER_MAP={mc_draws} | "
            f"sources={sorted(cfg['sources'])} | cost_mode={cfg['cost_mode']} | "
            f"use_floor_area={cfg['use_floor_area']} | fs_mode={cfg['fs_mode']}"
        )

        field_counter = 0
        for scenario, fields in found.items():
            if not fields:
                continue

            print(f"  [SCENARIO] {scenario} | {len(fields)} hazard fields")
            for hazard_field in fields:
                field_counter += 1
                print(f"  [PROGRESS] Hazard field {field_counter}/{total_fields}")

                df = process_field_decomposition(
                    hazard_field=hazard_field,
                    bld=bld,
                    scenario=scenario,
                    mode=mode,
                    mc_draws=mc_draws,
                    fs_thresholds=fs_thresholds,
                    out_draw_country_csv=out_country_draws,
                    out_draw_region_csv=out_region_draws,
                )
                if df is not None and not df.empty:
                    records.append(df)

        if not records:
            raise RuntimeError(f"No hazard fields produced any results for MODE={mode}")

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
