# UHI_STRUCTURE v1.1 — supplementary scripts for the manuscript

These scripts extend the v1.0.2 deposit. They read the two data records and the code deposit's
`data/inputs/` (built by `make_inputs.py`), plus the version-3 additions to the companion record.
Set the paths once (see `supplement_paths.py`):

```bash
export UHI_AIR_DATA=/path/to/air_record            # 10.5281/zenodo.22006932
export UHI_AIR_COMPANION=/path/to/companion_v3     # 10.5281/zenodo.22108287, v2 or later
export UHI_CODE_INPUTS=/path/to/UHI_STRUCTURE/data/inputs
export UHI_CODE_SCRIPTS=/path/to/UHI_STRUCTURE/scripts
export UHI_EXTRA_INPUTS=$UHI_AIR_COMPANION                  # where the v3 files live (defaults to ../inputs)
```

Run order is free; each script prints what it reproduces.

| script | reproduces | inputs beyond the records |
|---|---|---|
| `building_volume.py` | building-volume row of Table 1, Supplementary Table S6 (volume part), S1.6 | `gee_built_volume_extraction.csv`; `iso_cont.py` (ISO-3 → continent lookup) |
| `land_cover.py` | land-cover rows of Supplementary Table S6, S1.6 | `gee_ndvi_extraction_5km.csv` |
| `regional_gap.py` | Supplementary Table S2 (nine regional designs), S1.1, the room-to-grow split of S1.4. `UHI_VOLUME_SAMPLE=1` (default) uses the 1,106-city volume-joined primary sample the table was built on; `UHI_VOLUME_SAMPLE=0` uses all 1,108 cities | `gee_nightlights_extraction_cleaned.csv`, `ghs_wup_mtuc_r2025a_uc_stats_subset.csv`, `mtuc_overlap_crosswalk_*.csv`, `seasonal_uhi_panel*.csv` |
| `level_intervals.py` | 95 % intervals for the level quantities (zone slopes and means, distance bins, Supplementary Table S3), the Mundlak terms by element, the moving-boundary decomposition (Supplementary Table S5) and the seasonal level means of S1.7 | MTUC subset, crosswalks, `seasonal_uhi_panel.csv` |
| `panel_intervals.py` | Driscoll–Kraay and GHCN-M intervals, the balanced-panel epoch slopes and size×time interaction, the income-spline partial effects at the 90th and 99th percentiles (Supplementary Table S4) | none (uses `gdp_rcs.py` from the deposit) |
| `seasonal_table.py` | Table 2 of the main text, the seasonal rows of Supplementary Table S2 and the seasonal level means of S1.7 | `seasonal_uhi_panel*.csv` |
| `seasonal_within_city.py`, `seasonal_within_city_1975.py` | the seasonal city-epoch panels behind S1.7 and Table 2 of the main text | `seasonal_by_elem.csv` |
| `fetch_seasonal.py` | rebuilds `seasonal_by_elem.csv` from NOAA (streams ~11 GB of `.dly` files; not needed if the shipped file is used) | network access to NCEI |

`building_volume.py` is also imported (its first half) by `level_intervals.py`, `panel_intervals.py`, `land_cover.py`
and `regional_gap.py` to build the merged panel; run it once first to see the join coverage.

Requirements as for v1.0.2 (`numpy`, `pandas`, `scipy`, `statsmodels`, `linearmodels`), plus
`requests` for `fetch_seasonal.py` only.
