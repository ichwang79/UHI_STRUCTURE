# UHI_STRUCTURE

Analysis code for the empirical structure of the air-temperature urban heat island: how its level
scales across cities, how its trend moves within a city over time, and how the two separate by
element and by region.

The code reads two published data records and writes everything else. It contains no data of its
own beyond what it derives.

Each DOI below is a Zenodo concept DOI and resolves to the current version.

| | DOI | Licence |
|---|---|---|
| **This code** | [10.5281/zenodo.22109009](https://doi.org/10.5281/zenodo.22109009) | MIT |
| **Air-temperature UHI record** | [10.5281/zenodo.22006932](https://doi.org/10.5281/zenodo.22006932) | CC-BY-4.0 |
| **Companion city panels** | [10.5281/zenodo.22108287](https://doi.org/10.5281/zenodo.22108287) | CC-BY-4.0 |

## Getting the data

Download both records and point the scripts at them. They may live in one directory or two.

```bash
export UHI_AIR_DATA=/path/to/air_record
export UHI_AIR_COMPANION=/path/to/companion
```

Every script resolves a filename the same way — the air record first, then the companion — through
`scripts/uhi_paths.py`. Passing `--air` and `--companion` on the command line overrides the
environment. If both records are unpacked into one directory, set both variables to it.

Requirements: Python 3.9+, `numpy`, `pandas`, `scipy`, `statsmodels`, `linearmodels`, `matplotlib` (`requests` only for `fetch_seasonal.py`).

`si_robustness_suite.py` also reads the city groupings; point `UHI_AIR_GROUPINGS` at the companion record, which ships `city_groupings.csv`.

## Order of operations

`make_inputs.py` runs first and builds `data/inputs/` from the two records. Nothing is written
back into either record; all output lands in `data/inputs`, `data/results` and `figures`.

```bash
python scripts/make_inputs.py
```

It ends by re-estimating the size law and comparing against the reference values, and exits
non-zero if any of the three fits moves:

```
  daytime (TMAX)        -0.180     n=630
  mean (TAVG)           +0.216     n=866
  nighttime (TMIN)      +0.629     n=630
```

Then, in this order:

| script | what it builds |
|---|---|
| `within_city_panel.py` | the within-city epoch panel under all three rural-reference constructions |
| `within_city_panel_daynight.py` | the same panel split into nocturnal and daytime channels |
| `si_robustness_suite.py` | the full diagnostic suite on the within-city trend — 84 checks, ~15 min |
| `si_reference_sweeps.py` | the rural-annulus, screen-radius and lapse-rate sweeps |

## Analysis

Each reads `data/inputs/` and prints its results; the level-model scripts need only the inputs.

| script | question |
|---|---|
| `oke_analysis.py` | the population-size law, by element, region, climate zone and development phase |
| `mundlak_ladder.py` | between- and within-city density coefficients estimated jointly |
| `extreme_bounds.py` | which drivers survive an extreme-bounds search over covariate combinations |
| `gdp_rcs.py` | the functional form of the income term, by nested tests and AIC |
| `income_over_time.py` | how the income gradient moves across epochs |
| `rural_reference_longdiff.py` | the long-difference check on the rural reference |
| `yang2024_within_city.py` | the within-city response reproduced on an independent canopy-UHI product |


## Supplementary analyses added in version 1.1

These reproduce the Supplementary Information of the manuscript "Bigger cities have larger heat islands, denser ones do not, and growing ones get hotter" and read
the companion record (GEE extractions, the GHS-WUP-MTUC subset and crosswalks, and the
seasonal station-season file and panels). They resolve paths through `scripts/supplement_paths.py`
(`UHI_AIR_DATA`, `UHI_AIR_COMPANION`, optional `UHI_CODE_INPUTS`, `UHI_CODE_SCRIPTS`, `UHI_EXTRA_INPUTS`).

| script | reproduces |
|---|---|
| `building_volume.py` | building-volume row of Table 1 and Supplementary Table S6 (volume part) |
| `land_cover.py` | land-cover rows of Supplementary Table S6 |
| `regional_gap.py` | Supplementary Table S2 (the regional gap under nine designs) and the room-to-grow split of S1.4 |
| `level_intervals.py` | Supplementary Table S3, the Mundlak terms by element in Table 1, Supplementary Table S5, the seasonal level means of S1.7 |
| `panel_intervals.py` | the balanced-panel slopes and income-spline percentiles of Supplementary Table S4, the Driscoll–Kraay and GHCN-M intervals of Supplementary Table S1 |
| `seasonal_table.py` | Table 2 and the seasonal rows of Supplementary Table S2 |
| `seasonal_within_city.py`, `seasonal_within_city_1975.py` | the seasonal city-epoch panels from `seasonal_by_elem.csv` |
| `fetch_seasonal.py` | rebuilds `seasonal_by_elem.csv` from NOAA GHCN-Daily (about 11 GB streamed; the file ships with the companion record) |

Version 1.1 also drops three scripts the manuscript no longer uses (`density_shape_check.py`,
`seasonal_uhi.py`, `make_fig_replication.py`) and the satellite cross-check printout at the end of
`oke_analysis.py`; the numbers that mattered from them are in Supplementary Table S1.

## Figures

`figstyle.py` holds the shared plotting style and `uhi_paths.py` resolves deposited
filenames across the two records; neither is run directly.

| script | output |
|---|---|
| `make_main_figures.py` | the main display items |
| `make_fig3_regional.py` | the regional structure figure — needs `si_robustness_suite.py` first |

## What this cannot rebuild

Stated plainly, because a reader will otherwise look for it.

- **The raw daily archive.** GHCN-Daily `.dly` files are not redistributed by either record. The
  air record ships the station panel built from them, which is what these scripts read.
- **The YCEO rasters.** The companion record ships the extracted per-city panel, not the source
  GeoTIFFs, and documents how it was extracted. Re-extracting needs the rasters from the SEDAC
  product page.
- **The per-variant Yang et al. sweep.** `yang2024_within_city.py` runs its main table from
  the panel the companion record ships. The wide all-indicator table behind the supplementary
  sweep is not deposited; set `YANG_WIDE` if you have it, and the script skips that section
  otherwise.
- **The GHCN-M v4 monthly archive.** The companion record ships the derived cross-check panels;
  the monthly files come from NOAA/NCEI directly.
- **The rural-reference construction figure.** Its input table compares the three constructions
  across start years and is not produced by any script here; `si_reference_sweeps.py` and the
  `window` rows of `si_robustness_suite.py` cover the same ground numerically. The figure is
  not part of the submitted display items.

## Licence

MIT, see `LICENSE`. The data records carry their own licences at their own DOIs.
