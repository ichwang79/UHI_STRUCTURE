#!/usr/bin/env python3
"""
Did the income-UHI relationship change over 2000-2020?  (Fig. 3c-d)

Two questions, one panel each:

  (c) Cross-city gradient. At each five-year snapshot, regress a city's UHI on
      its log income, controlling city size and climate zone. A rising
      coefficient means income separates cities more sharply than it used to.

  (d) Trajectories by income tercile. Cities are assigned to terciles by their
      2020 income and each tercile's median UHI is tracked back through the
      snapshots, so the same cities are compared with themselves over time.

Snapshots are five-year windows centred on 2000, 2005, 2010, 2015 and 2020; a
city enters a snapshot only if its urban station has >=3 valid years in the
window and at least three rural references survive the same test.

UHI is rebuilt here rather than read from a table, because the snapshot windows
are narrower than those used elsewhere in the paper. The construction is the
paper's standard one: urban annual mean minus the median of its rural
references, with a 6.5 C/km lapse-rate correction for the urban-rural elevation
difference, and rural candidates dropped if they sit within 12 km of any other
city centroid (the contamination screen).

Inputs
------
  --release <dir>   the GHCN-Daily station UHI dataset release:
                   annual_by_elem.csv          station-year TAVG
                   need_broad_meta.csv         station lat/lon/elevation
                   city_station_match_broad.csv  city -> urban + rural stations
  data/inputs/uhi_panel_koppen_final_reconstructed.csv   income, population, Koppen
  data/inputs/city_centroids.csv                          centroids for the 12 km screen

Outputs
-------
  data/income_gradient_over_time.csv     year, n, coefficient, 95% CI, median GDPpc
  data/income_tercile_trajectory.csv     year x tercile median UHI

Both feed Fig. 3 panels c and d. The script checks itself against the published
values and exits non-zero on any drift.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from uhi_paths import find
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN = ROOT / "data" / "inputs"

SNAPSHOTS = {2000: (1998, 2002), 2005: (2003, 2007), 2010: (2008, 2012),
             2015: (2013, 2017), 2020: (2018, 2022)}
MIN_YEARS = 3          # valid years required inside a five-year window
MIN_RURAL = 3          # rural references required
SCREEN_KM = 12.0       # rural station must be this far from any other city
LAPSE = 6.5 / 1000     # C per metre

PUBLISHED_GRADIENT = {2000: 0.159, 2005: 0.209, 2010: 0.347, 2015: 0.426, 2020: 0.334}
PUBLISHED_TERCILE = {
    "lower-GDP":  {2000: 0.12, 2005: 0.13, 2010: 0.17, 2015: 0.06, 2020: 0.18},
    "mid-GDP":    {2000: 0.05, 2005: 0.10, 2010: 0.12, 2015: 0.07, 2020: 0.07},
    "higher-GDP": {2000: 0.28, 2005: 0.31, 2010: 0.41, 2015: 0.33, 2020: 0.33},
}
TERCILES = ["lower-GDP", "mid-GDP", "higher-GDP"]


def window_mean(series: dict, a: int, b: int) -> float:
    v = [series[y] for y in range(a, b + 1) if y in series]
    return float(np.mean(v)) if len(v) >= MIN_YEARS else np.nan


def build_uhi(p3: Path) -> pd.DataFrame:
    """City x snapshot UHI, rebuilt from station annual means."""
    meta = pd.read_csv(p3 / "need_broad_meta.csv", dtype={"id": str})
    S = {r.id: (r.lat, r.lon, r.elev) for r in meta.itertuples()}

    adf = pd.read_csv(p3 / "annual_by_elem.csv", dtype={"id": str})
    tavg: dict[str, dict[int, float]] = {}
    for r in adf.itertuples():
        if pd.notna(r.tavg):
            tavg.setdefault(r.id, {})[r.year] = r.tavg

    cent = pd.read_csv(find("city_centroids.csv", air=p3))
    proj = lambda la, lo: np.c_[np.asarray(la) * 111.0,
                                np.asarray(lo) * 111.0 * np.cos(np.radians(la))]
    tree = cKDTree(proj(cent.lat.values, cent.lon.values))
    clean_cache: dict[str, bool] = {}

    def uncontaminated(sid: str) -> bool:
        if sid not in clean_cache:
            la, lo, _ = S[sid]
            d, _ = tree.query(proj(np.array([la]), np.array([lo])))
            clean_cache[sid] = bool(d[0] >= SCREEN_KM)
        return clean_cache[sid]

    match = pd.read_csv(p3 / "city_station_match_broad.csv",
                        dtype={"urban": str, "rural": str})
    rows = []
    for r in match.itertuples():
        u = r.urban
        if u not in tavg:
            continue
        rural = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else [])
                 if s in tavg and uncontaminated(s)]
        if len(rural) < MIN_RURAL:
            continue
        u_elev = S[u][2]
        for year, (a, b) in SNAPSHOTS.items():
            uw = window_mean(tavg[u], a, b)
            if np.isnan(uw):
                continue
            valid = [s for s in rural if not np.isnan(window_mean(tavg[s], a, b))]
            if len(valid) < MIN_RURAL:
                continue
            r_elev = np.nanmedian([S[s][2] for s in valid])
            corr = LAPSE * (u_elev - r_elev) if np.isfinite(u_elev) and np.isfinite(r_elev) else 0.0
            rural_mean = np.median([window_mean(tavg[s], a, b) for s in valid])
            rows.append({"CityID": r.city_id, "year": year, "uhi": uw - rural_mean + corr})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", "--p3", dest="release",
                    default=(os.environ.get("UHI_AIR_DATA") or os.environ.get("UHI_RELEASE_DIR")
                             or os.environ.get("UHI_P3_DIR")),
                    help="path to the GHCN-Daily station UHI dataset release")
    args = ap.parse_args()
    if not args.release:
        ap.error("give --release <dir> (or set UHI_AIR_DATA)")
    p3 = Path(args.release).expanduser().resolve()

    dv = build_uhi(p3)
    k = pd.read_csv(IN / "uhi_panel_koppen_final_reconstructed.csv",
                    usecols=["CityID", "year", "ln_gdp_c", "pop_sum", "koppen_main_group"])
    k["lp"] = np.log10(k.pop_sum.clip(lower=1))
    d = dv.merge(k, on=["CityID", "year"], how="inner").dropna(subset=["uhi", "ln_gdp_c", "lp"])
    d["kop"] = d.koppen_main_group.fillna("C")

    # (c) cross-city income gradient at each snapshot
    grad = []
    for year in SNAPSHOTS:
        s = d[d.year == year]
        m = smf.ols("uhi ~ ln_gdp_c + lp + C(kop)", data=s).fit(cov_type="HC1")
        ci = m.conf_int().loc["ln_gdp_c"]
        grad.append(dict(year=year, n=len(s),
                         gradient=round(float(m.params["ln_gdp_c"]), 4),
                         lo=round(float(ci.iloc[0]), 4), hi=round(float(ci.iloc[1]), 4),
                         median_gdppc_k=round(float(np.exp(s.ln_gdp_c.median())), 2)))
    grad = pd.DataFrame(grad)
    grad.to_csv(ROOT / "data/income_gradient_over_time.csv", index=False)

    # (d) trajectories by 2020 income tercile
    g20 = d[d.year == 2020].set_index("CityID").ln_gdp_c
    d["tercile"] = d.CityID.map(pd.qcut(g20, 3, labels=TERCILES))
    traj = (d.dropna(subset=["tercile"])
              .groupby(["year", "tercile"], observed=True).uhi
              .agg(["median", "size"]).round(4).reset_index()
              .rename(columns={"median": "median_uhi", "size": "n"}))
    traj.to_csv(ROOT / "data/income_tercile_trajectory.csv", index=False)

    print(grad.to_string(index=False))
    print()
    print(traj.pivot(index="tercile", columns="year", values="median_uhi").to_string())

    drift = [f"gradient {r.year}: {r.gradient:+.3f} vs {PUBLISHED_GRADIENT[r.year]:+.3f}"
             for r in grad.itertuples()
             if abs(r.gradient - PUBLISHED_GRADIENT[r.year]) > 0.001]
    for t in TERCILES:
        for y, v in PUBLISHED_TERCILE[t].items():
            got = traj[(traj.tercile == t) & (traj.year == y)].median_uhi
            if len(got) and abs(float(got.iloc[0]) - v) > 0.005:
                drift.append(f"tercile {t} {y}: {float(got.iloc[0]):+.2f} vs {v:+.2f}")
    if drift:
        print("\nDRIFT from published values:")
        for x in drift:
            print("  " + x)
        return 1
    print("\nreproduces the published gradient and tercile values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
