# Arctic Permafrost Building Damage Model

This repository contains modeling frameworks for estimating
**mid-century building damages caused by permafrost degradation across
the Arctic circumpolar region**.

The models combine:

-   High-resolution **Arctic building datasets (HABITAT +
    OpenStreetMap)**
-   **Permafrost geotechnical hazard maps**
-   **Region-specific building replacement costs**
-   **Monte Carlo uncertainty propagation**

The workflow was developed to support the study:

**Refined modelling of Arctic circumpolar building stock increases
estimated mid-century permafrost degradation damages.**

------------------------------------------------------------------------

# Overview

Permafrost thaw can reduce the **bearing capacity of building
foundations**, resulting in structural damage or full replacement.

This repository provides two complementary modeling frameworks.

  -----------------------------------------------------------------------
  Model                               Purpose
  ----------------------------------- -----------------------------------
  Monte Carlo Damage Model            Estimate building damages with full
                                      uncertainty propagation

  Stepwise Decomposition Model        Quantify how improved exposure
                                      datasets amplify damage estimates
  -----------------------------------------------------------------------

Both models use **continuous bearing-capacity loss hazard maps derived
from CMIP6 climate projections**.

------------------------------------------------------------------------

# Repository Structure

    Permafrost-Infrastructure-Damage-Model
    │
    ├── damage_model.py
    ├── damage_model_decomposition.py
    ├── README.md
    │
    ├── data/
    │   ├── replacement_costs/
    │   └── hazard_maps/
    │
    ├── outputs/
    │   ├── damage_results/
    │   └── decomposition_results/
    │
    ├── figs/
        ├── uncertainty_sensitivity_plots.py
        ├── damage_amplification_map.py
        ├── decomposition_waterfall_plot.py
        └── README.md
    


Note: The **building inventory dataset is archived externally** (see
Data section below).

------------------------------------------------------------------------

# Model 1: Monte Carlo Damage Model

Script:

    damage_model.py

This script estimates **building damages from projected permafrost
bearing-capacity loss** while propagating multiple uncertainty sources.

### Key features

-   Monte Carlo simulation
-   Continuous hazard sampling from **netCDF BC loss maps**
-   Multiple uncertainty sources
-   Scenario-based analysis
-   Regional and country-level outputs

### Uncertainty sources

The model propagates uncertainty in:

-   Engineering **safety factors**
-   **Permafrost extent** beneath buildings
-   **Building detection completeness**
-   **Occupancy classification errors**
-   **Building height / story estimation**
-   Optional **hazard and exposure scaling**

Run modes include:

    ALL
    FS_ONLY
    EXTENT_ONLY
    DETECTION_ONLY
    TYPE_ONLY
    STORIES_ONLY

Each run produces **mean, minimum, and maximum damage estimates across
hazard maps**.

### Output files

The Monte Carlo model produces the following output files:

    BC_risk_analysis_v3_uncert_netcdf__ALL_mean.csv
    BC_risk_analysis_v3_uncert_netcdf__ALL_min.csv
    BC_risk_analysis_v3_uncert_netcdf__ALL_max.csv

Additional files contain **Monte Carlo draw-level outputs** used for
uncertainty analysis and figure generation.

------------------------------------------------------------------------

# Model 2: Exposure Decomposition Model

Script:

    damage_model_decomposition.py

This model evaluates how **progressively improved building exposure
datasets change damage estimates**.

### Exposure refinement blocks

  Block   Description
  ------- -------------------------------------------------------
  B1      OSM buildings only, 2D footprint area
  B2      OSM + HABITAT buildings
  B3      Occupancy-specific replacement costs
  B4      Residential floor area (stories) instead of footprint

This experiment isolates how improvements in **geospatial infrastructure
data** affect economic risk estimates.

------------------------------------------------------------------------

# Data

The modeling workflow relies on three primary datasets.

## Arctic building inventory

The Arctic building dataset used in this study is publicly archived at
the **Arctic Data Center**.

https://arcticdata.io/catalog/view/doi%3A10.18739%2FA21R6N311

This dataset contains the **HABITAT-OSM circumpolar building
inventory**, combining:

-   Deep learning building detections from high-resolution satellite
    imagery (HABITAT)
-   Manually digitized OpenStreetMap building footprints

Primary input file used by the model:

    HABITAT_OSM_bldg_type_activity_topo_with_stories.gpkg

Users should download this dataset from the Arctic Data Center and
update the file path in the scripts accordingly.

------------------------------------------------------------------------

## Permafrost hazard maps

NetCDF grids representing projected **bearing-capacity loss**.

These bearing capacity maps are part of ongoing **U.S. National Science
Foundation (NSF) Grants RISE-2019691 and 2022504** and will become
publicly available following completion of those projects (expected fall
2026).

The datasets can currently be obtained upon reasonable request from:

Dmitry Streletskiy\
strelets@gwu.edu

Hazard maps contain the variable:

    bc_change

Supported climate scenarios:

    SSP2-4.5
    SSP5-8.5

------------------------------------------------------------------------

## Replacement cost inventory

The replacement cost dataset contains regional estimates of building
replacement cost per unit area.

Fields include:

    shapeGroup
    shapeName
    RES_COST_PER_AREA
    NONRES_COST_PER_AREA
    COMBINED_COST_PER_AREA

Costs are converted to **PPP USD 2024** during the analysis.

------------------------------------------------------------------------

# Figure Generation

Scripts used to reproduce figures from the manuscript are located in:

    figs/

  -----------------------------------------------------------------------
  Script                              Description
  ----------------------------------- -----------------------------------
  uncertainty_sensitivity_plots.py    Generates uncertainty and
                                      sensitivity analysis plots

  damage_amplification_map.py         Produces Arctic maps showing
                                      regional damage amplification

  decomposition_waterfall_plot.py     Creates exposure decomposition
                                      waterfall plots
  -----------------------------------------------------------------------

See `figs/README.md` for figure reproduction instructions.

------------------------------------------------------------------------

# Running the Models

## Install dependencies

Typical Python environment:

    python >= 3.9
    numpy
    pandas
    geopandas
    xarray

Install using:

    pip install numpy pandas geopandas xarray

------------------------------------------------------------------------

## Configure file paths

Scripts require the following paths to be configured:

    HAZARD_NC_DIR
    BUILDINGS_GPKG
    COST_CSV
    OUT_DIR

------------------------------------------------------------------------

## Run the model

Monte Carlo uncertainty model:

    python damage_model.py

Exposure decomposition experiment:

    python damage_model_decomposition.py

------------------------------------------------------------------------

# Output

Model outputs are written as CSV tables summarizing damages by:

-   Scenario
-   Country
-   Administrative region
-   Data source
-   Building activity status

Metrics include:

    total_area
    total_cost
    damaged_area
    damaged_cost

All monetary values are reported in **PPP USD 2024**.

------------------------------------------------------------------------

# Citation

If you use this repository or the building dataset, please cite:

Manos et al.\
*Refined modelling of Arctic circumpolar building stock increases
estimated mid-century permafrost degradation damages.*

Building dataset:

Manos et al.\
**HABITAT-OSM Arctic building inventory**\
Arctic Data Center\
https://arcticdata.io/catalog/view/doi%3A10.18739%2FA21R6N311

------------------------------------------------------------------------

# License

This repository is released under an open research license.
