#!/usr/bin/env python3
"""
Does income enter the between-city UHI model, and in what shape?  (Fig. 3a-b)

A linear income term is easy to reject, but that is a statement about the
functional form, not about income. This script asks the question in every
plausible form -- linear, quadratic, cubic, and a four-knot restricted cubic
spline -- always on top of the same size + climate baseline, and reports what
each form adds. The spline is then traced out as a partial effect: the UHI of a
city at a given income relative to a city at mean income, holding size and
climate zone fixed.

The spline is Harrell's four-knot restricted cubic basis with knots at the 5th,
35th, 65th and 95th percentiles of centred log income. Restricting the tails to
be linear is what keeps the high-income end from being driven by a handful of
cities; the upturn reported in the paper survives that restriction.

Confidence bands are a nonparametric city bootstrap (default 1000 draws): the
spline is refitted on each resample and the 2.5th and 97.5th percentiles of the
partial effect are taken pointwise. Bootstrapping rather than using the delta
method matters here because the curve is a nonlinear function of four
coefficients whose errors are correlated.

Input :  data/inputs/uhi_level_model_cities.csv
Output:  data/gdp_functional_forms.csv      what each functional form adds
         data/gdp_rcs_partial_effect.csv    the fitted curve with bootstrap band
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "data/inputs/uhi_level_model_cities.csv"

KNOT_PCT = [5, 35, 65, 95]
DVS = [("uhi", "mean (TAVG)"), ("uhi_tmin", "night (TMIN)")]
GRID = 120


def rcs_basis(g, knots):
    """Harrell four-knot restricted cubic spline basis (two terms beyond linear)."""
    t1, t2, t3, t4 = knots
    D = (t4 - t1) ** 2
    cp = lambda x, a: np.where(np.asarray(x) - a > 0, (np.asarray(x) - a) ** 3, 0.0)
    r2 = (cp(g, t1) - cp(g, t3) * (t4 - t1) / (t4 - t3) + cp(g, t4) * (t3 - t1) / (t4 - t3)) / D
    r3 = (cp(g, t2) - cp(g, t3) * (t4 - t2) / (t4 - t3) + cp(g, t4) * (t3 - t2) / (t4 - t3)) / D
    return r2, r3


def partial_effect(params, g, knots):
    """Spline contribution at g, expressed relative to a mean-income city (g = 0)."""
    r2, r3 = rcs_basis(g, knots)
    z2, z3 = rcs_basis(np.array([0.0]), knots)
    f = params["g"] * g + params["g_rcs2"] * r2 + params["g_rcs3"] * r3
    f0 = params["g_rcs2"] * z2[0] + params["g_rcs3"] * z3[0]
    return f - f0


def prepare(d):
    d = d.copy()
    d["kop"] = pd.Categorical(d.kop, categories=["C", "A", "B", "D"])
    d["g"] = d.ln_gdp - d.ln_gdp.mean()
    knots = np.nanpercentile(d.g, KNOT_PCT)
    d["g_rcs2"], d["g_rcs3"] = rcs_basis(d.g.values, knots)
    d["g2"], d["g3"] = d.g ** 2, d.g ** 3
    return d, knots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    raw = pd.read_csv(SRC)
    d, knots = prepare(raw)
    gdp_mean = raw.ln_gdp.mean()
    rng = np.random.default_rng(args.seed)

    forms, curves = [], []
    for dv, label in DVS:
        s = d.dropna(subset=[dv, "ln_pop", "g"])
        base = smf.ols(f"{dv} ~ ln_pop + C(kop)", data=s).fit()

        # (1) what does each functional form add over size + climate?
        for name, terms in [("linear", "g"), ("quadratic", "g + g2"),
                            ("cubic", "g + g2 + g3"),
                            ("RCS (4 knots)", "g + g_rcs2 + g_rcs3")]:
            m = smf.ols(f"{dv} ~ ln_pop + C(kop) + {terms}", data=s).fit(cov_type="HC1")
            ft = m.f_test(" , ".join(f"{t} = 0" for t in terms.replace(" ", "").split("+")))
            forms.append(dict(dv=label, n=len(s), form=name,
                              r2=round(m.rsquared, 4),
                              delta_r2=round(m.rsquared - base.rsquared, 4),
                              aic=round(m.aic, 1),
                              joint_p=round(float(np.ravel(ft.pvalue)), 4),
                              ln_pop=round(float(m.params["ln_pop"]), 4),
                              ln_pop_p=float(m.pvalues["ln_pop"])))

        # (2) trace the spline out, with a city bootstrap band
        fit = smf.ols(f"{dv} ~ ln_pop + C(kop) + g + g_rcs2 + g_rcs3", data=s).fit(cov_type="HC1")
        gs = np.linspace(s.g.quantile(.01), s.g.quantile(.99), GRID)
        point = partial_effect(fit.params, gs, knots)

        draws = np.empty((args.boot, GRID))
        idx = np.arange(len(s))
        for b in range(args.boot):
            bs = s.iloc[rng.choice(idx, len(s), replace=True)]
            try:
                fb = smf.ols(f"{dv} ~ ln_pop + C(kop) + g + g_rcs2 + g_rcs3", data=bs).fit()
                draws[b] = partial_effect(fb.params, gs, knots)
            except Exception:
                draws[b] = np.nan
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)

        curves.append(pd.DataFrame(dict(dv=label, n=len(s), ln_gdp=gs + gdp_mean,
                                        effect=point, lo=lo, hi=hi)))
        print(f"{label}: n={len(s)}, curve from {point[0]:+.2f} to {point[-1]:+.2f} C")

    pd.DataFrame(forms).to_csv(ROOT / "data/gdp_functional_forms.csv", index=False)
    out = pd.concat(curves, ignore_index=True).round(4)
    out.to_csv(ROOT / "data/gdp_rcs_partial_effect.csv", index=False)

    # the income distribution behind the curve, for the figure's rug/histogram
    raw[["ln_gdp"]].to_csv(ROOT / "data/gdp_distribution.csv", index=False)
    print(f"knots (centred ln GDPpc): {np.round(knots, 3)}; mean ln GDPpc = {gdp_mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
