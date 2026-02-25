
# Arctic Permafrost Bearing Capacity Damage Model

This repository contains two modeling frameworks for estimating Arctic building damages 
from projected bearing capacity (BC) loss under mid-century climate scenarios.

---

# 1. Model Components

## 1.1 Monte Carlo Uncertainty Model

Script:
```
damage_model.py
```

Features:
- Full Monte Carlo uncertainty propagation
- One-at-a-time (OAT) sensitivity modes
- Hazard-pooled summaries (mean / min / max)
- Residential and non-residential damage splits

---

## 1.2 Stepwise Decomposition Model

Script:
```
damage_model_decomposition.py
```

Features:
- Incremental exposure refinement blocks (B1–B4)
- Consistent FS sampling across blocks
- Designed for damage amplification attribution

---

# 2. Required Inputs

## 2.1 Hazard Data (netCDF)

Set:
```
HAZARD_NC_DIR
```

Each `.nc` file must:

- Contain variable: `bc_change`
- Contain coordinates: `lon`, `lat`
- Include scenario identifier in filename:
  - `ssp245`
  - `ssp585`

BC values must represent fractional loss (0–1).
Percent values (0–100) are automatically scaled.
Negative values are flipped if interpreted as decreases.

---

## 2.2 Building Inventory (GeoPackage)

Set:
```
BUILDINGS_GPKG
BUILDINGS_LAYER = "buildings"
```

### Required Columns

| Column        | Description                                  |
|--------------|----------------------------------------------|
| shapeGroup   | Country code (USA, CAN, RUS)                |
| shapeName    | Region name                                  |
| Source       | OSM or HABITAT                               |
| Value        | 1 = residential                              |
| Shape_Area   | Footprint area (m²)                          |
| EXTENT       | C / D / S / I damage class                   |
| ntl_status   | ACTIVE / not                                 |
| num_stories  | Optional for floor area expansion             |

---

## 2.3 Cost Inventory (CSV)

Set:
```
COST_CSV
```

### Required Columns

| Column                    | Description |
|---------------------------|-------------|
| shapeGroup                | Country code |
| shapeName                 | Region name |
| RES_COST_PER_AREA         | Residential replacement cost per m² |
| NONRES_COST_PER_AREA      | Non-residential replacement cost per m² |
| COMBINED_COST_PER_AREA    | Used in decomposition blocks B1–B2 |

All costs are treated as nominal and converted to PPP USD 2024 using:

- `GDP_DEFLATOR_2021_TO_2024` (uncertainty model)
- `PPP_INFLATION_FACTOR_2024` (decomposition model)

---

# 3. Core Parameters

## 3.1 Scenario Matching

Hazard files are matched using:

- SSP245
- SSP585

---

## 3.2 Safety Factor (FS)

Ranges:

- USA / CAN: 2.5–3.0
- RUS: 1.05–1.56

Threshold is computed as:

```
threshold = 1 − (1 / FS)
```

---

## 3.3 EXTENT Multipliers

| EXTENT | Multiplier Range |
|--------|------------------|
| C      | 0.90–1.00 |
| D      | 0.50–0.90 |
| S      | 0.10–0.50 |
| I      | 0.00 |

---

# 4. Monte Carlo Uncertainty Configuration

Defined in:

```
MODES_TO_RUN
MODE_FLAGS
MC_DRAWS_BY_MODE
```

Available modes:

- ALL
- FS_ONLY
- EXTENT_ONLY
- DETECTION_ONLY
- TYPE_ONLY
- STORIES_ONLY

Key uncertainty parameters:

- FS_RANGES
- EXTENT_BOUNDS
- HABITAT_DET_F1
- TYPE_F1_RES
- TYPE_F1_NONRES
- DELTA_STORY_VALUES
- HAZARD_SIGMA
- AREA_SIGMA

Randomness is controlled via:

```
RANDOM_SEED
```

---

# 5. Decomposition Blocks

Defined in:

```
MODE_CFG
```

| Block | Description |
|-------|------------|
| B1 | OSM only, 2D, combined region cost |
| B2 | OSM + HABITAT, 2D, combined region cost |
| B3 | Add occupancy-specific costs |
| B4 | Add floor area expansion |

All blocks use identical FS draws for comparability.

---

# 6. Outputs

For each mode:

```
<OUT_BASE>__<MODE>_mean.csv
<OUT_BASE>__<MODE>_min.csv
<OUT_BASE>__<MODE>_max.csv
```

Optional per-draw outputs:

```
<OUT_BASE>__<MODE>_MC_country_draws.csv
<OUT_BASE>__<MODE>_MC_region_draws.csv
```

Outputs include:

- Total area (m²)
- Total replacement cost (PPP USD 2024)
- Damaged area
- Damaged cost
- Residential / non-residential splits
- Proportion damaged

---

# 7. Running the Model

Edit configuration variables at the top of each script:

```
HAZARD_NC_DIR
BUILDINGS_GPKG
COST_CSV
OUT_DIR
```

Then execute:

```
python damage_model.py
python damage_model_decomposition.py
```

---

# 8. Units

- Area: m²
- Cost: PPP USD 2024
- Loss: fraction (0–1)

---

End of README.
