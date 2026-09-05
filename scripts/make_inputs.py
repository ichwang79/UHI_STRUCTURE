#!/usr/bin/env python3
"""
make_inputs.py — assemble the analysis inputs from the two public data records.

The analysis scripts in this folder read from ``../data/inputs/``. That directory is not
distributed (the inputs are derived products of two separate data releases); this script
rebuilds it, then verifies the rebuild by re-estimating the published size-law table.

Sources
-------
1. GHCN-Daily station UHI dataset release  (deposited; see Data & code)
     annual_by_elem.csv                  station-level annual TMIN/TMAX/TAVG
     city_station_match_longrecord.csv   long-record city-station matching
     city_station_match_broad.csv        2000-2020 city-station matching
     longrecord_city_uhi.csv             full-record city UHI (tavg/tmin/tmax)
     city_population.csv                 city population and sampling strata
     need_stations_meta.csv              long-record station metadata (incl. elevation)
     city_covariates.csv                 analysis-ready per-city covariates
     city_predictors_panel.csv           CityID x year GHS predictors (density/GDP/built/pop)
     hist_predictors.csv                 historical GHS predictors, 1975-2020 epochs,
                                         including the precomputed GDP spline basis g1-g3

2. Co-located satellite LST panel  (deposited separately; see Data & code)
     city_lst_panel.csv                  day/night LST, elevation, NDVI, wind, coastal
                                         distance and the corrected surface UHI, on the
                                         same GHS-UCDB cities as the station record

The records are self-contained: everything the analysis needs ships with them, so --scidata
is optional and retained only for rebuilding city_predictors_panel.csv from source.

Usage
-----
    python make_inputs.py --p3 /path/to/release
Paths may also be set with the environment variables ``UHI_P3_DIR`` / ``UHI_SCIDATA_DIR``.
Add ``--verify-only`` to check an existing data/inputs/ without rebuilding.

Verification
------------
The rebuilt inputs must reproduce Table "oke_size_law_fits.csv" exactly:
    daytime (TMAX)    -0.180     mean (TAVG)   +0.216     nighttime (TMIN)  +0.629
Any deviation is reported and the script exits non-zero.
"""
from __future__ import annotations
import argparse
import os, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data" / "inputs"
AUX = ROOT / "data" / "aux" / "city_lst_aux.csv"

PUBLISHED = {"daytime (TMAX)": -0.180, "mean (TAVG)": 0.216, "nighttime (TMIN)": 0.629}
PUBLISHED_N = {"daytime (TMAX)": 630, "mean (TAVG)": 866, "nighttime (TMIN)": 630}

# The inputs come from two deposits. They may sit in one directory (unpack both there) or in two.
AIR_FILES = ["annual_by_elem.csv", "city_station_match_longrecord.csv", "city_uhi_epoch_panel.csv",
             "longrecord_city_uhi.csv", "city_population.csv", "need_stations_meta.csv"]
COMPANION_FILES = ["city_predictors_panel.csv", "hist_predictors.csv", "city_covariates.csv"]
SD_FILES = []   # kept so --scidata stays accepted; nothing is required from it


class _Sources:
    """Resolves a filename against the air deposit first, then the companion deposit."""

    def __init__(self, air: Path, companion: Path):
        self.air, self.companion = air, companion

    def __truediv__(self, name: str) -> Path:
        p = self.air / name
        return p if p.exists() else self.companion / name


def resolve(args) -> tuple[_Sources, Path]:
    air = Path(args.p3 or os.environ.get("UHI_AIR_DATA")
               or os.environ.get("UHI_P3_DIR", "")).expanduser()
    comp = Path(args.companion or os.environ.get("UHI_AIR_COMPANION", "")).expanduser() \
        if (args.companion or os.environ.get("UHI_AIR_COMPANION")) else air
    sd = Path(args.scidata or os.environ.get("UHI_SCIDATA_DIR", "")).expanduser()
    missing = []
    for label, base, files in (("--air", air, AIR_FILES), ("--companion", comp, COMPANION_FILES)):
        if not base or not base.is_dir():
            missing.append(f"{label} not set or not a directory: {base!s}")
            continue
        for f in files:
            if not (base / f).exists():
                missing.append(f"{label}: missing {f}")
    if missing:
        sys.exit("Cannot locate the data deposits:\n  " + "\n  ".join(missing) +
                 "\n\nPass --air and --companion, or set UHI_AIR_DATA and UHI_AIR_COMPANION.\n"
                 "Both may be the same directory if the two deposits were unpacked together.")
    return _Sources(air, comp), sd


EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
LAPSE, SCREEN_KM, MIN_YEARS, MIN_RURAL = 6.5 / 1000.0, 12.0, 3, 3


def build_anomaly_panel(p3: Path) -> pd.DataFrame:
    """Epoch-level station UHI with an anomaly rural reference; see section 6 of build()."""
    from scipy.spatial import cKDTree
    meta = pd.read_csv(p3 / "need_broad_meta.csv", dtype={"id": str})
    S = {r.id: r.elev for r in meta.itertuples()}
    cen = pd.read_csv(p3 / "city_centroids.csv")[["lon", "lat"]].drop_duplicates()
    proj = lambda la, lo: np.c_[la * 111.0, lo * 111.0 * np.cos(np.radians(la))]
    dist, _ = cKDTree(proj(cen.lat.values, cen.lon.values)).query(proj(meta.lat.values, meta.lon.values))
    keep = set(meta.id[dist >= SCREEN_KM])

    adf = pd.read_csv(p3 / "annual_by_elem.csv", dtype={"id": str})
    TA: dict = {}
    for r in adf.itertuples():
        if pd.notna(r.tavg):
            TA.setdefault(r.id, {})[r.year] = r.tavg
    match = pd.read_csv(p3 / "city_station_match_broad.csv", dtype={"urban": str, "rural": str})

    cache: dict = {}
    def wm(sid, y):
        k = (sid, y)
        if k not in cache:
            d = TA.get(sid, {})
            x = [d[t] for t in range(y - 2, y + 3) if t in d]
            cache[k] = float(np.mean(x)) if len(x) >= MIN_YEARS else np.nan
        return cache[k]

    rows = []
    for r in match.itertuples():
        u = r.urban
        if u not in TA or u not in S:
            continue
        cl = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else [])
              if s in TA and s in keep and s in S]
        if len(cl) < MIN_RURAL:
            continue
        yrs = [y for y in EPOCHS if np.isfinite(wm(u, y))]
        if not yrs:
            continue
        obs = [(si, ti, wm(st, y)) for si, st in enumerate(cl)
               for ti, y in enumerate(yrs) if np.isfinite(wm(st, y))]
        if not obs:
            continue
        ns, nt = len(cl), len(yrs)
        X = np.zeros((len(obs), ns + nt - 1)); v = np.zeros(len(obs))
        for i, (si, ti, val) in enumerate(obs):
            X[i, si] = 1.0
            if ti > 0:
                X[i, ns + ti - 1] = 1.0
            v[i] = val
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        alpha, tau = beta[:ns], np.r_[0.0, beta[ns:]]
        cnt: dict = {}
        for _, ti, _ in obs:
            cnt[ti] = cnt.get(ti, 0) + 1
        base = float(np.median(alpha))
        zr = np.nanmedian([S[s] for s in cl])
        ec = LAPSE * (S[u] - zr) if np.isfinite(S[u]) and np.isfinite(zr) else 0.0
        for ti, y in enumerate(yrs):
            if cnt.get(ti, 0) >= MIN_RURAL:
                rows.append({"CityID": r.city_id, "year": y, "n_ref": len(cl),
                             "uhi_obs": wm(u, y) - (base + tau[ti]) + ec})
    return pd.DataFrame(rows)


def _lst_panel(p3: Path) -> Path:
    """The satellite panel is deposited separately; UHI_LST_PANEL overrides its location."""
    f = Path(os.environ.get("UHI_LST_PANEL", p3 / "city_lst_panel.csv"))
    if not f.exists():
        raise SystemExit(
            f"satellite LST panel not found at {f}\n"
            "It is deposited separately from the station record; download it and set\n"
            "UHI_LST_PANEL=/path/to/city_lst_panel.csv")
    return f


def build(p3: Path, sd: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. city_uhi_predictors.csv -------------------------------------------------
    uhi = pd.read_csv(p3 / "longrecord_city_uhi.csv")
    # analysis-ready per-city covariates ship with the release; fall back to the
    # panel mean over 2000-2020 only if that file is absent.
    cov = p3 / "city_covariates.csv"
    if cov.exists():
        pred = (pd.read_csv(cov)
                  .dropna(subset=["CityID"]).drop_duplicates("CityID")
                  .set_index("CityID")[["ln_popdensity", "ln_gdp_c", "frac_urban_built"]])
    else:
        panel = pd.read_csv(p3 / "city_predictors_panel.csv",
                            usecols=["CityID", "year", "ln_popdensity", "ln_gdp_c",
                                     "frac_urban_built"])
        pred = panel.groupby("CityID")[["ln_popdensity", "ln_gdp_c", "frac_urban_built"]].mean()
    out = (uhi[["city_id", "uhi_tavg", "uhi_tmin", "country", "continent", "CityID", "koppen"]]
           .merge(pred.reset_index(), on="CityID", how="left"))
    out = out[["city_id", "uhi_tavg", "uhi_tmin", "country", "continent", "CityID",
               "ln_popdensity", "ln_gdp_c", "frac_urban_built", "koppen"]]
    out.to_csv(OUT / "city_uhi_predictors.csv", index=False)

    # ---- 2. representativeness.csv --------------------------------------------------
    pop = pd.read_csv(p3 / "city_population.csv")
    rep = pop.copy()
    if "koppen_main_group" not in rep.columns:
        rep = rep.merge(uhi[["CityID", "koppen", "country", "continent"]].drop_duplicates("CityID"),
                        on="CityID", how="left").rename(columns={"koppen": "koppen_main_group"})
    rep.to_csv(OUT / "representativeness.csv", index=False)

    # ---- 2b. derived city tables carried straight through from the deposits ---------
    # These are inputs to the level model and the cross-instrument check. They are distributed rather
    # than rebuilt here because they carry ancillary columns (NDVI, wind, coastal distance,
    # elevation and their z-scores) assembled from the satellite panel.
    for name in ("uhi_level_model_cities.csv", "yceo_city_panel.csv"):
        src = p3 / name
        if Path(src).exists():
            pd.read_csv(src).to_csv(OUT / name, index=False)
        else:
            print(f"  note: {name} not found in the deposits; "
                  "the level-model and cross-instrument scripts will not run without it")

    # ---- 3. city_station_match.csv (long-record) ------------------------------------
    match = pd.read_csv(p3 / "city_station_match_longrecord.csv",
                        dtype={"urban": str, "rural": str})
    match.to_csv(OUT / "city_station_match.csv", index=False)

    # ---- 4. annual_tavg.csv ---------------------------------------------------------
    ann = pd.read_csv(p3 / "annual_by_elem.csv", dtype={"id": str})
    ann[["id", "year", "tavg"]].dropna(subset=["tavg"]).to_csv(OUT / "annual_tavg.csv", index=False)

    # ---- 5. uhi_panel_koppen_final_reconstructed.csv --------------------------------
    # full satellite/LST panel with the raw components the analysis reads (elevation, NDVI,
    # wind, coastal distance, day/night LST) — shipped with the release.
    full = pd.read_csv(_lst_panel(p3))
    full = full.rename(columns={"koppen_climate_group": "koppen_main_group",
                                "UHI_corrected_global_pooled": "UHI_corrected",
                                "UHI_raw_global_pooled": "UHI_raw"})
    full.to_csv(OUT / "uhi_panel_koppen_final_reconstructed.csv", index=False)

    # ---- 6. hist_stage3_panel.csv ---------------------------------------------------
    # The epoch-level station UHI, joined to the GHS-UCDB predictors.
    #
    # The released city_uhi_epoch_panel.csv takes its rural reference from whichever of a
    # city's clean-rural stations report in each epoch's window. The set therefore changes
    # between epochs, and a station entering or leaving moves the median, so a within-city
    # change in UHI mixes a real change with a change in who the reference is. Over twenty
    # years the set barely turns over and the drift is negligible; over forty-five it turns
    # over almost completely and the two constructions correlate at only r = 0.54, with
    # opposite signs. The trend half of the analysis is estimated on that within variation, so the
    # panel is rebuilt here with an anomaly reference instead: tau is fitted from
    # T[s,t] = alpha_s + tau_t over ALL of the city's clean-rural stations, and the reference
    # is R(t) = median_s(alpha_s) + tau_t. Nothing is discarded, the profile no longer moves
    # when the set turns over, and the median matches the level convention of the
    # median-of-stations estimator it replaces.
    #
    # The released construction is kept alongside it as hist_stage3_panel_varying.csv so the
    # difference stays visible. Both are built from the release alone.
    # hist_predictors.csv is required, not optional. Its frac_built is the GHS built-up SURFACE
    # share (mean ~0.19); the satellite panel's frac_urban_built is the MODIS urban-class share
    # (mean ~0.81), and within a city the two correlate -0.26. They are different measurements of
    # built-up land, not one variable under two names, so the within-city trend model cannot take
    # whichever happens to be present.
    if not (p3 / "hist_predictors.csv").exists():
        raise FileNotFoundError(
            f"{p3 / 'hist_predictors.csv'} is missing. It carries frac_built, the GHS built-up "
            "surface share the within-city trend model controls for. Do not substitute "
            "frac_urban_built (the MODIS urban-class share): it is a different measurement. "
            "Fetch hist_predictors.csv from the companion deposit.")
    hp = pd.read_csv(p3 / "hist_predictors.csv")
    if "frac_built" not in hp.columns:
        raise RuntimeError("hist_predictors.csv does not carry frac_built; see the note above.")

    def _attach(dv):
        h = dv.merge(hp, on=["CityID", "year"], how="left")
        # The GDP spline basis ships with the release; regenerating it from knot percentiles
        # reproduces it only to ~3 decimals, which is enough to move a p-value in the fourth.
        have = {"g1", "g2", "g3"} <= set(h.columns) and h[["g1", "g2", "g3"]].notna().any().all()
        g = h["ln_gdp_c"].to_numpy(dtype=float)
        if np.isfinite(g).any() and not have:            # Harrell 4-knot RCS basis
            t1, t2, t3, t4 = np.nanpercentile(g[np.isfinite(g)], [5, 35, 65, 95])
            den = (t4 - t1) ** 2
            cp = lambda a: np.where(g - a > 0, (g - a) ** 3, 0.0)
            h["g1"] = g
            h["g2"] = (cp(t1) - cp(t3) * (t4 - t1) / (t4 - t3) + cp(t4) * (t3 - t1) / (t4 - t3)) / den
            h["g3"] = (cp(t2) - cp(t3) * (t4 - t2) / (t4 - t3) + cp(t4) * (t3 - t2) / (t4 - t3)) / den
        return h

    _attach(pd.read_csv(p3 / "city_uhi_epoch_panel.csv")).to_csv(
        OUT / "hist_stage3_panel_varying.csv", index=False)
    _attach(build_anomaly_panel(p3)).to_csv(OUT / "hist_stage3_panel.csv", index=False)

    print(f"wrote {len(list(OUT.glob('*.csv')))} input files to {OUT}")


def verify() -> int:
    """re-estimate the published size-law table from the assembled inputs"""
    import statsmodels.formula.api as smf

    pr = pd.read_csv(OUT / "city_uhi_predictors.csv")
    pr["uhi_max"] = 2 * pr.uhi_tavg - pr.uhi_tmin
    rep = pd.read_csv(OUT / "representativeness.csv")
    keep = [c for c in ["CityID", "pop", "abslat", "income", "latband", "kop", "sampled"]
            if c in rep.columns]
    d = pr.merge(rep[keep], on="CityID", how="left")
    # the published level sample is the cities carrying analysis covariates; without this
    # restriction the fit runs on ~37 extra cities and none of the three slopes match
    d = d.dropna(subset=["ln_popdensity"])
    d = d[d["pop"] > 0].copy()
    d["lp"] = np.log10(d["pop"])

    print("\nverification — Oke size law (country-clustered SE)")
    print(f"  {'measure':18} {'rebuilt':>9} {'published':>10} {'n':>6} {'n_pub':>6}  status")
    bad = 0
    for y, lab in [("uhi_max", "daytime (TMAX)"), ("uhi_tavg", "mean (TAVG)"),
                   ("uhi_tmin", "nighttime (TMIN)")]:
        s = d.dropna(subset=[y, "lp", "country"])
        g = pd.factorize(s.country)[0]
        m = (smf.ols(f"{y} ~ lp", data=s).fit(cov_type="cluster", cov_kwds={"groups": g})
             if s.country.nunique() > 1 else smf.ols(f"{y} ~ lp", data=s).fit(cov_type="HC1"))
        b, n = m.params["lp"], int(m.nobs)
        ok = abs(b - PUBLISHED[lab]) < 0.0006 and n == PUBLISHED_N[lab]
        bad += (not ok)
        print(f"  {lab:18} {b:+9.3f} {PUBLISHED[lab]:+10.3f} {n:6d} {PUBLISHED_N[lab]:6d}  "
              f"{'OK' if ok else 'MISMATCH'}")
    if bad:
        print(f"\n{bad} of 3 fits do not match the published table.")
    else:
        print("\nall fits reproduce the published table exactly.")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--air", "--release", "--p3", dest="p3",
                    help="the air-temperature UHI data deposit")
    ap.add_argument("--companion", default=None,
                    help="the companion data deposit (defaults to --air)")
    ap.add_argument("--scidata", help="deprecated; accepted and ignored")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify an existing data/inputs/ without rebuilding")
    args = ap.parse_args()

    if not args.verify_only:
        p3, sd = resolve(args)
        build(p3, sd)
    elif not OUT.exists():
        sys.exit(f"nothing to verify: {OUT} does not exist")

    sys.exit(1 if verify() else 0)


if __name__ == "__main__":
    main()
