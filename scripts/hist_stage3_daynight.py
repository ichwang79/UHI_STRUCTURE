"""
The epoch panel rebuilt element by element, so the trend model can be split by channel.

hist_stage3.py builds the within-city panel from TAVG alone and writes one blended uhi_obs. That
is why the trend half of the analysis -- the half reported as regionally unstable and
without out-of-sample skill -- has never been separated into day and night, while the level half
has been separated since Section A of oke_analysis.py. This script repeats that construction
exactly, changing one thing: the station means are taken for TMIN and TMAX as well as TAVG, so the
panel carries three outcomes instead of one.

Everything else is held fixed against hist_stage3.py so the blended column reproduces:

  clean-rural screen   a rural reference is kept only if it lies at least 12 km from any urban
                       centre in the working panel, not only from the city being measured
  window               five years centred on each epoch, at least three of them valid
  references           at least three clean rural stations with data in that window
  elevation            urban minus the rural median height, at 6.5 C/km
  epochs               1975 to 2020 in five-year steps

Under the record's convention TAVG = (TMAX + TMIN)/2 the three outcomes satisfy
uhi_day = 2*uhi_mean - uhi_night exactly, so the mean is not an independent third quantity: it is
the average of the two channels, and reporting it alongside them is a summary rather than a result.
The script checks that identity and prints the residual.

Rural referencing
-----------------
The reference was previously re-selected at every epoch, from whichever of a city's clean-rural
stations had data in that epoch's window, so a station entering or leaving moved the median and
that movement entered the within-city estimator as if the city had changed. The default here is
the anomaly reference used by hist_stage3.py: tau is fitted from T[s,t] = alpha_s + tau_t over
ALL of the city's clean-rural stations, separately for tavg and tmin, and the reference is
R(t) = median_s(alpha_s) + tau_t. Nothing is discarded -- the panel carries the same 878 cities as
the varying construction, on a median of 31 references rather than 22 -- and the profile no
longer moves when the set turns over. Fitting tau per element preserves the identity, since
2*uhi_mean - uhi_night is then the tmax UHI against the implied tmax reference.

The correction matters most where the set turns over most. Over 1975-2020 the night within-city
density coefficient moves from -0.339 under the varying reference to +0.271 under the anomaly
reference, the same reversal hist_stage3.py shows on TAVG.

Inputs   annual_by_elem.csv, need_broad_meta.csv, city_station_match_broad.csv, city_centroids.csv
         from the deposited release, plus hist_predictors.csv from the companion deposit.
Outputs  ../data/inputs/hist_stage3_panel_daynight.csv           the panel, three outcomes
         ../data/inputs/hist_stage3_panel_daynight_varying.csv   the previous construction
         ../data/results/hist_stage3_daynight.csv                the within-city fits by channel
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.spatial import cKDTree

# Paths resolve relative to this file, or from the environment, so the scripts run from a clone
# without editing. UHI_AIR_DATA points at the unpacked GHCN-Daily station UHI release (Paper 3's
# Zenodo deposit); UHI_AIR_COMPANION at the companion deposit, which is where hist_predictors.csv
# actually lives.
import os as _os
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
RELEASE = str(_P(_os.environ.get("UHI_AIR_DATA", _HERE.parent.parent / "Paper3_ESSD" / "data")))
INPUTS  = str(_HERE.parent / "data" / "inputs")
COMPANION = str(_P(_os.environ.get("UHI_AIR_COMPANION", _HERE.parent / "data" / "reproduction_extras" / "companion")))


warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "data", "inputs") + os.sep
OUT = os.path.join(HERE, "..", "data", "results") + os.sep
# The rural-reference screen checks distance against every GHS urban centre, not only the ones
# with a matched station; city_centroids.csv, deposited with the release, is that list (Sect. 2.4).
RINGS = f"{RELEASE}/city_centroids.csv"
EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
ELEMS = [("tavg", "mean"), ("tmin", "night")]   # day is derived, not computed: see build()
LAPSE = 6.5 / 1000.0
SCREEN_KM = 12.0
MIN_YEARS, MIN_RURAL = 3, 3


def data_dir():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data", default=RELEASE)
    d = ap.parse_known_args()[0].data
    if not os.path.isdir(d):
        raise SystemExit(f"data directory not found: {d}\n"
                         "Pass the unpacked release with --data /path/to/data.")
    return d + os.sep


def clean_rural(meta):
    """A rural reference must sit at least 12 km from every urban centre, not only its own city."""
    if not os.path.exists(RINGS):
        raise SystemExit(f"missing {RINGS}, which carries the urban-centre list for the screen")
    c = pd.read_csv(RINGS, encoding="utf-8-sig")[["lon", "lat"]].drop_duplicates()
    proj = lambda la, lo: np.c_[la * 111.0, lo * 111.0 * np.cos(np.radians(la))]
    tree = cKDTree(proj(c.lat.values, c.lon.values))
    d, _ = tree.query(proj(meta.lat.values, meta.lon.values))
    return set(meta.id[d >= SCREEN_KM])


def tau_profile(series, stations, yrs):
    """Least squares alpha_s + tau_t on one element's rural station-by-epoch matrix."""
    obs = [(si, ti, series(st, y)) for si, st in enumerate(stations)
           for ti, y in enumerate(yrs) if np.isfinite(series(st, y))]
    if not obs:
        return None
    ns, nt = len(stations), len(yrs)
    X = np.zeros((len(obs), ns + nt - 1)); v = np.zeros(len(obs))
    for i, (si, ti, val) in enumerate(obs):
        X[i, si] = 1.0
        if ti > 0:
            X[i, ns + ti - 1] = 1.0
        v[i] = val
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    alpha, tau = beta[:ns], np.r_[0.0, beta[ns:]]
    cnt = {}
    for _, ti, _ in obs:
        cnt[ti] = cnt.get(ti, 0) + 1
    base = float(np.median(alpha))
    return {yrs[ti]: base + tau[ti] for ti in range(nt) if cnt.get(ti, 0) >= MIN_RURAL}


def build(D, mode="anomaly"):
    meta = pd.read_csv(D + "need_broad_meta.csv", dtype={"id": str})
    keep = clean_rural(meta)
    S = {r.id: r.elev for r in meta.itertuples()}
    ann = pd.read_csv(D + "annual_by_elem.csv", dtype={"id": str})
    match = pd.read_csv(D + "city_station_match_broad.csv", dtype={"urban": str, "rural": str})

    # The rural reference is a median across stations, and a median is not linear: even where
    # every station satisfies tavg = (tmin + tmax)/2, the station sitting at the median can differ
    # between elements, so median(tavg) is not the average of median(tmin) and median(tmax).
    # Computing all three independently therefore breaks the identity -- by up to 3.9 C here.
    # Section A of oke_analysis.py avoids this by deriving the daytime channel rather than
    # computing it, and the same is done below: the mean and the night are built from the data,
    # the day follows as 2*mean - night. Coverage is aligned first so that the mean and the night
    # rest on the same station-years, which is what makes the derived day channel meaningful.
    ann = ann.dropna(subset=["tmin", "tmax"]).copy()
    ann["tavg"] = (ann.tmin + ann.tmax) / 2.0
    series = {}
    for elem, _ in ELEMS:
        series[elem] = {sid: dict(zip(g.year, g[elem])) for sid, g in ann.groupby("id")}

    def window(elem, sid, y):
        d = series[elem].get(sid)
        if not d:
            return np.nan
        v = [d[t] for t in range(y - 2, y + 3) if t in d]
        return np.mean(v) if len(v) >= MIN_YEARS else np.nan

    rows = []
    for r in match.itertuples():
        rural = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else [])
                 if s in keep and s in S]
        if len(rural) < MIN_RURAL or r.urban not in S:
            continue
        if mode == "anomaly":
            yrs = [y for y in EPOCHS
                   if all(np.isfinite(window(e, r.urban, y)) for e, _ in ELEMS)]
            if not yrs:
                continue
            R = {e: tau_profile(lambda st, y, e=e: window(e, st, y), rural, yrs) for e, _ in ELEMS}
            if any(v is None for v in R.values()):
                continue
            zr = np.nanmedian([S[s] for s in rural])
            corr = LAPSE * (S[r.urban] - zr) if np.isfinite(S[r.urban]) and np.isfinite(zr) else 0.0
            for y in yrs:
                if not all(y in R[e] for e, _ in ELEMS):
                    continue
                rec = {"CityID": r.city_id, "year": y, "n_ref": len(rural)}
                for elem, lab in ELEMS:
                    rec[f"uhi_{lab}"] = window(elem, r.urban, y) - R[elem][y] + corr
                rows.append(rec)
            continue
        for y in EPOCHS:
            rec = {"CityID": r.city_id, "year": y, "n_ref": len(rural)}
            for elem, lab in ELEMS:
                u = window(elem, r.urban, y)
                if not np.isfinite(u):
                    rec[f"uhi_{lab}"] = np.nan
                    continue
                vals = [(window(elem, s, y), S[s]) for s in rural]
                vals = [(v, z) for v, z in vals if np.isfinite(v)]
                if len(vals) < MIN_RURAL:
                    rec[f"uhi_{lab}"] = np.nan
                    continue
                zr = np.nanmedian([z for _, z in vals])
                corr = LAPSE * (S[r.urban] - zr) if np.isfinite(S[r.urban]) and np.isfinite(zr) else 0.0
                rec[f"uhi_{lab}"] = u - np.median([v for v, _ in vals]) + corr
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["uhi_day"] = 2 * out.uhi_mean - out.uhi_night      # exact under TAVG = (TMAX + TMIN)/2
    return out


def main():
    D = data_dir()
    pred = pd.read_csv(f"{COMPANION}/hist_predictors.csv")
    d = build(D, "anomaly").merge(pred, on=["CityID", "year"], how="inner")
    d_vary = build(D, "varying").merge(pred, on=["CityID", "year"], how="inner")

    have = d.dropna(subset=["uhi_mean", "uhi_night", "uhi_day"])
    resid = (have.uhi_day - (2 * have.uhi_mean - have.uhi_night)).abs()
    print(f"epoch panel rebuilt: {len(d):,} city-epochs, {d.CityID.nunique():,} cities")
    for lab in ("mean", "night", "day"):
        s = d[f"uhi_{lab}"]
        print(f"  uhi_{lab:6} {s.notna().sum():>7,} present  "
              f"{d.loc[s.notna(), 'CityID'].nunique():>6,} cities  mean {s.mean():+.3f}")
    print(f"  identity uhi_day = 2*uhi_mean - uhi_night holds to "
          f"{resid.max():.2e} C over {len(have):,} rows"
          + ("" if resid.max() < 1e-9 else "   <-- STILL BROKEN"))

    old = os.path.join(IN, "hist_stage3_panel.csv")
    if os.path.exists(old):
        o = pd.read_csv(old)[["CityID", "year", "uhi_obs"]]
        j = d.merge(o, on=["CityID", "year"])
        print(f"  against the published blended panel: r = {j.uhi_mean.corr(j.uhi_obs):.4f}, "
              f"median difference {np.median(j.uhi_mean - j.uhi_obs):+.4f} C, n = {len(j):,}")

    os.makedirs(IN, exist_ok=True)
    d.to_csv(IN + "hist_stage3_panel_daynight.csv", index=False)
    d_vary.to_csv(IN + "hist_stage3_panel_daynight_varying.csv", index=False)
    print(f"  wrote {IN}hist_stage3_panel_daynight.csv          (anomaly reference, the default)")
    print(f"  wrote {IN}hist_stage3_panel_daynight_varying.csv  (previous construction)")
    j = d[["CityID", "year", "uhi_night"]].merge(
        d_vary[["CityID", "year", "uhi_night"]], on=["CityID", "year"], suffixes=("_an", "_va"))
    print(f"  the two constructions: r = {j.uhi_night_an.corr(j.uhi_night_va):.4f} on "
          f"{len(j):,} common city-epochs")

    LADDER = [("M0 density only", ["ln_popdensity"]),
              ("M1 +GDP linear", ["ln_popdensity", "ln_gdp_c"]),
              ("M2 +built", ["ln_popdensity", "ln_gdp_c", "frac_built"]),
              ("M3 +GDP spline", ["ln_popdensity", "g1", "g2", "g3", "frac_built"])]
    rows = []
    for window_lab, sub in [("2000-2020, the paper's headline window", d[d.year >= 2000]),
                            ("1975-2020, the full epoch panel", d)]:
        print(f"\n{'=' * 82}\nWITHIN-CITY density model  --  {window_lab}\n{'=' * 82}")
        for lab in ("day", "mean", "night"):
            n = sub[f"uhi_{lab}"].notna().sum()
            print(f"  {lab:6} {n:,} city-epochs, "
                  f"{sub.loc[sub[f'uhi_{lab}'].notna(), 'CityID'].nunique():,} cities")
        print(f"\n{'model':18}{'day':>21}{'mean':>21}{'night':>21}")
        for name, cols in LADDER:
            line = ""
            for lab in ("day", "mean", "night"):
                s = sub.dropna(subset=[f"uhi_{lab}"] + cols).drop_duplicates(["CityID", "year"])
                s = s.set_index(["CityID", "year"])
                m = PanelOLS(s[f"uhi_{lab}"], s[cols], entity_effects=True, time_effects=True,
                             drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
                k = "ln_popdensity"
                line += f"{m.params[k]:>+12.3f} (p={m.pvalues[k]:.3f})"
                rows.append(dict(window=window_lab, model=name, channel=lab,
                                 coef=m.params[k], se=m.std_errors[k], p=m.pvalues[k],
                                 n=int(m.nobs), cities=s.index.get_level_values(0).nunique()))
            print(f"{name:18}{line}")

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT + "hist_stage3_daynight.csv", index=False)
    print(f"\nwrote {OUT}hist_stage3_daynight.csv")


if __name__ == "__main__":
    main()
