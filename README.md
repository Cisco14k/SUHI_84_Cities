# Urban tree cooling with distance: P2 analysis code (Heat6 pipeline)

This is the Python code of the P2 study of the Heat6 series. It estimates how far green space, tree canopy and water
cool urban land-surface temperature in 84 city-regions, using Landsat Collection 2 Level-2 data. All inputs are open
data. The pipeline downloads and processes the data on one processing computer, and the synthesis and figures run
afterwards on the per-area result tables.

## Layout

| Folder | What it does | Where it runs |
|---|---|---|
| `01_download/` | Downloads and grids every input on the 120 m Landsat grid of each city-region: warm-season Landsat composites (`a42_landsat.py`), every clear warm-season Landsat scene 2013-2025 (`a42_scenes.py`), ESA WorldCover 2021, Copernicus GLO-30 DEM and JRC surface water (`a42_static.py`), GHSL R2023A built-up surface, volume and population (`a42_ghsl.py`), ETH canopy height 2020 (`a42_chm.py`), Köppen-Geiger classes (`a42_koppen.py`), NASA POWER daily climate (`a42_power.py`) and hourly ERA5 overpass forcing (`a42_era5h.py`). `a42_supervisor.py` starts and restarts the workers, and `a42_jobs.py` is the job queue that runs the analysis next to the data. `areas_h6.json` is the registry of study areas (P2 areas have `"paper": "P2"`). | Processing PC |
| `02_analysis_2013_2025/` | The per-area analysis chain (`run_all.py`, with `H6_PAPER=P2`): `prep.py` (analysis grids and Landsat back-casting of impervious and canopy fractions), `heat.py` (lapse-rate-adjusted surface temperature relative to the rural background), `seb.py` (surface energy balance at overpass), `causal.py`, `cooling.py` (ring exposures, composite ring DML, InVEST urban-cooling heat-mitigation index), `coolprob.py` (scene-level ring DML for every clear scene 2013-2025), `dml.py` (partially linear DML with spatial-block cross-fitting and block-clustered errors), `climate.py`, `aero.py`, `figpack.py`. | Processing PC |
| `03_extension_1985_2025/` | Extends P2 to 1985-2025. Two more composites per city-region (c. 1987 and 2013-2016; `h6_landsat_p2long.py`), every clear warm-season scene 1985-2012 (`h6_scenes_p2long.py`), ERA5 forcing for the new composites (`h6_era5h_p2long.py`), and a Landsat 7 / Landsat 8 overlap check (`h6_scenes_l7ov.py`). `ana_p2long/p2long_run.py` reruns the chain for four epochs, and `coolprob_long.py` estimates every scene with the structure of the nearest epoch. `p2long_extras.py` adds the canopy-threshold variants and residents within reach. The jobs in `ana_jobs/11x_*.json` run it through the job queue. | Processing PC |
| `04_synthesis_and_figures/` | Pools the scene-level estimates and makes the figures: `p2_common.py` (loading, DerSimonian-Laird pooling, reach definitions), `p2_rev_a.py` (magnitude, day-to-day variation, frequency F_delta, reaches, robustness values), `p2_rev_b.py` (weather modulation), `p2_rev_c.py`/`p2_rev_d.py` (persistence, confident-cooling days), `p2_rev_e.py`/`p2_rev_f.py` (InVEST variants and forecasts in held-out cities with city-bootstrap intervals), `p2_rev_g.py` (Paule-Mandel and REML checks), `p2_rv_composite.py`/`p2_rvbench.py` (omitted-confounder bounds), `p2_sens*.py` (distance sensitivity), `p2_long.py` (1985-2025 synthesis, interpolated reach with city bootstrap), `p2_fig*.py`, `p2_ed*.py`, `p2_maps.py`, `p2_figs_long.py`, `figstyle.py`. | Any computer |

## Running

1. **Data and per-area analysis (processing PC, Windows).** Put `01_download/` and `02_analysis_2013_2025/` (as the
   subfolder `ana/`) in one pipeline folder. Run `START_HEAT6.bat`: it creates the download environment, and
   `ana_setup.py` creates the analysis environment `ana_venv`. Downloads resume after interruptions. The per-area
   chain starts for each area as soon as its inputs are complete and writes `ana_out/<paper>/tables/*.csv`.
2. **1985-2025 extension.** Copy the contents of `03_extension_1985_2025/` into the same pipeline folder and run
   `START_P2_1985_2025.bat`. The run is finished when `ana_out/P2long/FINAL_DONE.json` exists. The results are in
   `ana_out/P2long/results_final.zip`.
3. **Synthesis and figures.** Unpack the result tables into `work/` below the folder you run from: `work/P2/tables`,
   `work/P2work`, `work/P2long/tables`, and `work/p2keys.json` (the list of P2 area keys). Then run the scripts from the
   package root, for example `python 04_synthesis_and_figures/p2_rev_a.py`. Outputs are written to `work/rev/`.
   `p2_fig1_long.py` also needs the design schematic as `work/inputs/published_fig1.png`, and `p2_figs_long.py` needs
   `work/inputs/p2_citynames.json`.

Paths on the processing PC come from `A42_PC_ROOT` (the pipeline folder). For code tests, set `A42_TEST_ROOT`.

## Environment

Python 3.11. See `requirements.txt` for the packages; record exact versions with `pip freeze` in the environment that
ran the analysis. Landsat, WorldCover and the DEM are read from the Microsoft Planetary Computer STAC API, which needs
no account. Open-Meteo, NASA POWER, figshare and the JRC servers are reached over public HTTP APIs.

## Data sources

- Landsat Collection 2 Level-2 (USGS):
  - Landsat 4-5 TM: https://doi.org/10.5066/P9IAXOVV
  - Landsat 7 ETM+: https://doi.org/10.5066/P9C7I13B
  - Landsat 8-9 OLI/TIRS: https://doi.org/10.5066/P9OGBGM6
- ESA WorldCover 2021 v200: https://doi.org/10.5281/zenodo.7254221
- Copernicus GLO-30 DEM: https://doi.org/10.5270/ESA-c5d3d65
- JRC Global Surface Water, GHSL R2023A, ETH Global Canopy Height 2020
- Köppen-Geiger 1991-2020 (Beck et al. 2023, figshare 21789074)
- NASA POWER daily, ERA5 via the Open-Meteo archive API

## Licence

MIT (see `LICENSE`).
