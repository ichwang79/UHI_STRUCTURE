#!/usr/bin/env python3
"""
Extreme-bounds analysis of the between-city UHI level model (Fig. 2e, Table 2).

Question: of the candidate drivers of a city's UHI level, which survive being
asked in every reasonable way? Rather than report one preferred specification,
we estimate the same relationship under all 384 combinations of modelling
choices and report, for each driver, the distribution of its coefficient and
the share of specifications *in which that driver appears* that find it
significant at 5%.

The sweep is the cross-product of:
  * which size/density term anchors the model  (density only, size only, both)
  * whether Koppen climate-zone fixed effects are included                (2)
  * which subset of the six remaining covariates is included    (2^6 = 64)
        -> 3 x 2 x 64 = 384 specifications total

A driver appears in 256 of the 384 if it is an anchor (absent only from the
"density only" or "size only" branch that excludes it), or 192 if it is a free
covariate. The reported percentiles and significance shares are always over
the specifications that actually contain that driver, not over all 384 --
"significant in 100% of 384 specifications" is imprecise for exactly this
reason and should read "100% of the 256 (or 192) specifications in which it
appears."

All predictors are standardised, so coefficients are °C per standard deviation
and are directly comparable across drivers. Standard errors are
heteroskedasticity-robust (HC1).

Run for both dependent variables used in the paper -- annual mean (`uhi`) and
nocturnal (`uhi_tmin`) -- because they are not interchangeable: the nocturnal
sample is smaller (670 v. 948 cities) and, in general, a robustness share
established on one element is not evidence about the other. Each element's
own extreme-bounds table must be deposited and cited separately.

Input :  data/inputs/uhi_level_model_cities.csv   (948 cities, one row each)
Output:  data/eba_uhi_level.csv        (mean/TAVG,  Fig. 2e; --dv uhi)
         data/eba_uhi_level_tmin.csv   (night/TMIN, §4.2, night; --dv uhi_tmin)
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/inputs/uhi_level_model_cities.csv"
OUT = {"uhi": ROOT / "data/eba_uhi_level.csv",
       "uhi_tmin": ROOT / "data/eba_uhi_level_tmin.csv"}
PRETTY_DV = {"uhi": "annual mean (TAVG)", "uhi_tmin": "nocturnal (TMIN)"}

ANCHORS = [["ln_density"], ["ln_pop"], ["ln_density", "ln_pop"]]
FREE = ["ln_gdp", "built", "ndvi_contrast", "wind", "abs_lat", "ln_coastal"]
CAND = ["ln_pop", "ln_density"] + FREE

LABEL = {
    "ln_pop": "City size (ln pop)",
    "ln_density": "Pop density (ln)",
    "ln_gdp": "GDP/capita (ln)",
    "built": "Built-up fraction",
    "ndvi_contrast": "Veg contrast (r-u)",
    "wind": "Wind",
    "abs_lat": "|Latitude|",
    "ln_coastal": "Coastal distance (ln)",
}


def run(dv: str, published: dict | None = None) -> pd.DataFrame:
    d = pd.read_csv(SRC)
    for c in CAND:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    d = d.dropna(subset=[dv] + [c + "_z" for c in CAND])

    draws = {c: [] for c in CAND}
    n_spec = 0
    for anchor in ANCHORS:
        for kop in (True, False):
            for r in range(len(FREE) + 1):
                for sub in itertools.combinations(FREE, r):
                    cols = anchor + list(sub)
                    f = f"{dv} ~ " + " + ".join(c + "_z" for c in cols)
                    if kop:
                        f += " + C(kop)"
                    m = smf.ols(f, data=d).fit(cov_type="HC1")
                    n_spec += 1
                    for c in cols:
                        draws[c].append((m.params[c + "_z"], m.pvalues[c + "_z"]))

    rows = []
    for c in CAND:
        b = np.array([x[0] for x in draws[c]])
        p = np.array([x[1] for x in draws[c]])
        lo, hi = np.percentile(b, [2.5, 97.5])
        pct_sig = round(100 * float((p < 0.05).mean()))
        sign_stab = round(100 * float(max((b > 0).mean(), (b < 0).mean())))
        rows.append(dict(
            driver=LABEL[c],
            median_beta=round(float(np.median(b)), 4),
            lo=round(float(lo), 4), hi=round(float(hi), 4),
            pct_sig=pct_sig, sign_stability=sign_stab,
            # robust = significant nearly always, sign never flips, bounds exclude zero
            robust=bool(pct_sig >= 90 and sign_stab >= 95 and (lo > 0) == (hi > 0)),
            n_specs=len(b),
        ))
    out = pd.DataFrame(rows).sort_values("pct_sig", ascending=False)
    out.to_csv(OUT[dv], index=False)

    print(f"=== {PRETTY_DV[dv]}: {len(d)} cities, {n_spec} specifications total ===")
    print(out.to_string(index=False))
    print("(robust = significant in >=90% of the specs it appears in, sign stable >=95%, bounds exclude 0)")

    # how the population terms behave depending on which of them anchors the model
    print("population terms by anchor:")
    for lbl, anchor in [("density alone", ["ln_density"]), ("size alone", ["ln_pop"]),
                        ("both together", ["ln_density", "ln_pop"])]:
        acc = {c: [] for c in anchor}
        for kop in (True, False):
            for r in range(len(FREE) + 1):
                for sub in itertools.combinations(FREE, r):
                    cols = anchor + list(sub)
                    f = f"{dv} ~ " + " + ".join(x + "_z" for x in cols) + (" + C(kop)" if kop else "")
                    m = smf.ols(f, data=d).fit(cov_type="HC1")
                    for c in anchor:
                        acc[c].append((m.params[c + "_z"], m.pvalues[c + "_z"]))
        for c in anchor:
            b = np.array([x[0] for x in acc[c]]); p = np.array([x[1] for x in acc[c]])
            print(f"  {lbl:15} {LABEL[c]:22} median β={np.median(b):+.3f}  "
                  f"significant in {100 * (p < .05).mean():.0f}%")

    if published:
        bad = [(r.driver, r.pct_sig, published[r.driver])
               for r in out.itertuples() if r.driver in published and r.pct_sig != published[r.driver]]
        print("reproduces published shares" if not bad else f"DRIFT: {bad}")
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dv", choices=["uhi", "uhi_tmin", "both"], default="both",
                    help="dependent variable: uhi (mean), uhi_tmin (nocturnal), or both (default)")
    args = ap.parse_args()

    # the published mean-UHI shares; a mismatch means the input or sweep has drifted
    published_mean = {"City size (ln pop)": 100, "Pop density (ln)": 55, "Wind": 50,
                      "Veg contrast (r-u)": 43, "GDP/capita (ln)": 18, "|Latitude|": 8,
                      "Built-up fraction": 0, "Coastal distance (ln)": 0}

    dvs = ["uhi", "uhi_tmin"] if args.dv == "both" else [args.dv]
    for dv in dvs:
        run(dv, published=published_mean if dv == "uhi" else None)


if __name__ == "__main__":
    main()
