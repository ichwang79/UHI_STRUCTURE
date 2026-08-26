#!/usr/bin/env python3
"""
rural_reference_longdiff.py — the evidence behind Figure 2 and Section 4.1.

The heat-island change between two periods is a difference of two urban-minus-rural gaps. The
urban side is the same thermometer at both ends. The rural side is a median over whichever of a
city's clean-rural stations reported in each period, and that set turns over. Any movement in it
is subtracted from the city and read as a change in the city.

This script measures how large that is, by building the same long difference twice and comparing
them city by city.

  varying   the reference is the median over whoever reports at each endpoint, which is the
            conventional construction
  fixed     only stations valid at BOTH endpoints are used, and the same set is used at both, so
            composition is constant by construction

Everything else is held identical between the two: the 12 km contamination screen, five-year
endpoint windows requiring three valid years, at least three rural references, 6.5 C/km on the
urban-minus-rural-median elevation, and the day channel derived as 2*mean - night rather than
computed.

Two details are not cosmetic. Station coverage is aligned to years carrying both TMIN and TMAX and
the mean is then formed as their average, because a rural median taken independently per element
need not sit on the same station and would break the day identity. And the comparison is reported
on the night channel, which is where the urban signal sits (Section 4.4); the day channel is
reported beside it and behaves the same way.

Three intervals are run, because the whole point is that the problem grows with the length of the
difference: 20 years, 35 years and 45 years.

Outputs, written to ../data/inputs/ for make_fig_rural_reference.py to read:

    ld20.csv, ld35.csv, ld45.csv   per-city changes under both constructions, plus the density
                                   change dX and the two reference counts
    reference_turnover.csv         median references reporting at one end against usable at both,
                                   by interval length; panel B of Figure 2

Run:
    python3 rural_reference_longdiff.py --data /path/to/release
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from uhi_paths import find
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "inputs"
LAPSE = 6.5 / 1000.0
SCREEN_KM = 12.0
MIN_YEARS = 3
MIN_RURAL = 3
INTERVALS = [(20, [2000, 2005], [2015, 2020]),
             (35, [1985, 1990], [2015, 2020]),
             (45, [1975, 1980], [2015, 2020])]


def project(lat, lon):
    lat = np.asarray(lat, float)
    return np.c_[lat * 111.0, np.asarray(lon, float) * 111.0 * np.cos(np.radians(lat))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("UHI_AIR_DATA"),
                   help="unpacked GHCN-Daily station UHI release")
    a = p.parse_args()
    if not a.data or not Path(a.data).is_dir():
        raise SystemExit("pass the unpacked release with --data /path/to/data")
    D = Path(a.data)

    meta = pd.read_csv(D / "need_broad_meta.csv", dtype={"id": str})
    S = dict(zip(meta.id, meta.elev))
    # the urban-centre list ships with the release; the screen is applied against all of it,
    # not only the cities that appear in this record
    centres = pd.read_csv(D / "city_centroids.csv")
    d, _ = cKDTree(project(centres.lat.values, centres.lon.values)).query(
        project(meta.lat.values, meta.lon.values))
    keep = set(meta.id.values[d >= SCREEN_KM])

    ann = pd.read_csv(D / "annual_by_elem.csv", dtype={"id": str}).dropna(subset=["tmin", "tmax"])
    ann["tavg"] = (ann.tmin + ann.tmax) / 2.0     # aligned coverage, so the day identity holds
    series = {e: {sid: dict(zip(g.year, g[e])) for sid, g in ann.groupby("id")}
              for e in ("tavg", "tmin")}
    match = pd.read_csv(D / "city_station_match_broad.csv", dtype={"urban": str, "rural": str})
    pred = pd.read_csv(find("hist_predictors.csv", air=D))

    def win(elem, sid, y):
        s = series[elem].get(sid)
        if not s:
            return np.nan
        v = [s[t] for t in range(y - 2, y + 3) if t in s]
        return float(np.mean(v)) if len(v) >= MIN_YEARS else np.nan

    def endpoint(urban, pool, epochs):
        out = {}
        for elem, lab in (("tavg", "mean"), ("tmin", "night")):
            vals = []
            for y in epochs:
                u = win(elem, urban, y)
                if not np.isfinite(u):
                    continue
                rv = [(win(elem, s, y), S.get(s, np.nan)) for s in pool]
                rv = [(v, z) for v, z in rv if np.isfinite(v)]
                if len(rv) < MIN_RURAL:
                    continue
                zr = np.nanmedian([z for _, z in rv])
                zu = S.get(urban, np.nan)
                ec = LAPSE * (zu - zr) if np.isfinite(zu) and np.isfinite(zr) else 0.0
                vals.append(u - np.median([v for v, _ in rv]) + ec)
            out[lab] = float(np.mean(vals)) if vals else np.nan
        out["day"] = 2 * out["mean"] - out["night"]
        return out

    def valid_at(elem, sid, epochs):
        return any(np.isfinite(win(elem, sid, y)) for y in epochs)

    turnover = []
    for span, early, late in INTERVALS:
        rows = []
        for r in match.itertuples():
            rural = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else [])
                     if s in keep and s in S]
            if len(rural) < MIN_RURAL or r.urban not in S:
                continue
            fixed = [s for s in rural
                     if all(valid_at(e, s, w) for e in ("tmin", "tavg") for w in (early, late))]
            rec = {"CityID": r.city_id, "country": r.country,
                   "n_rural_all": len(rural), "n_rural_fixed": len(fixed)}
            av, bv = endpoint(r.urban, rural, early), endpoint(r.urban, rural, late)
            for k in ("night", "day"):
                rec[f"vary_{k}"] = bv[k] - av[k]
            if len(fixed) >= MIN_RURAL:
                af, bf = endpoint(r.urban, fixed, early), endpoint(r.urban, fixed, late)
                for k in ("night", "day"):
                    rec[f"fix_{k}"] = bf[k] - af[k]
            rows.append(rec)
        d = pd.DataFrame(rows)
        xa = pred[pred.year.isin(early)].groupby("CityID").ln_popdensity.mean()
        xb = pred[pred.year.isin(late)].groupby("CityID").ln_popdensity.mean()
        d = d.merge((xb - xa).rename("dX").reset_index(), on="CityID",
                    how="inner").dropna(subset=["dX"])
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT / f"ld{span}.csv", index=False)

        both = d.dropna(subset=["vary_night", "fix_night", "dX"])
        turnover.append({"years": span, "all": int(d.n_rural_all.median()),
                         "common": int(d.n_rural_fixed.median()), "cities": len(both)})
        print(f"\n{span}-year difference   {early} -> {late}")
        print(f"  cities {len(d):,}, of which {len(both):,} admit a fixed reference")
        print(f"  references: median {d.n_rural_all.median():.0f} reporting at one end, "
              f"{d.n_rural_fixed.median():.0f} usable at both")
        for ch in ("night", "day"):
            fv = smf.ols(f"vary_{ch} ~ dX", data=both).fit(
                cov_type="cluster", cov_kwds={"groups": both.country})
            ff = smf.ols(f"fix_{ch} ~ dX", data=both).fit(
                cov_type="cluster", cov_kwds={"groups": both.country})
            print(f"    {ch:<6} varying {fv.params['dX']:+.3f} (p={fv.pvalues['dX']:.3f})   "
                  f"fixed {ff.params['dX']:+.3f} (p={ff.pvalues['dX']:.3f})   "
                  f"r = {both[f'vary_{ch}'].corr(both[f'fix_{ch}']):.3f}")
    pd.DataFrame(turnover).to_csv(OUT / "reference_turnover.csv", index=False)
    print(f"\nwrote ld20.csv, ld35.csv, ld45.csv and reference_turnover.csv to {OUT}")


if __name__ == "__main__":
    main()
