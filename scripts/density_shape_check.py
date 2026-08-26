"""
Does population density enter linearly? Tested on the sample the level model uses.

The paper bends the income term and leaves density straight. The reason for bending income --
that the relationship need not be monotone in the covariate -- applies to density too, and nothing
in the paper tests the restriction. This script does, at each of the three places density carries
an argument:

  level      the between-city cross-section, where density is negative once size is held fixed
  trend      the within-city panel, where densification raises UHI
  Mundlak    the joint model whose equality test the paper uses to separate the two

Density enters as a four-knot restricted cubic spline, placed as Harrell places them, and the
curvature terms are tested jointly against zero. Where they are indistinguishable from zero the
straight line is the right summary and nothing changes; where they are not, a single slope is
describing a curve, and the fitted shape is printed so it can be read.

A spline needs the covariate to move. Between cities density spans orders of magnitude, so it is
well identified there. Within a city it moves very little over a five-epoch window, so the panel
spline is reported alongside the share of variation it has to work with, and a curvature that
cannot be resolved is reported as that rather than as evidence of linearity.

Run:  python density_shape_check.py
Out:  ../data/results/density_shape_check.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "data", "inputs") + os.sep
OUT = os.path.join(HERE, "..", "data", "results") + os.sep
KNOT_PCT = [5, 35, 65, 95]
Q = [10, 25, 50, 75, 90]


def basis(x, k):
    x = np.asarray(x, float)
    t1, tk, tkm = k[0], k[-1], k[-2]
    cols = [x]
    for j in range(len(k) - 2):
        tj = k[j]
        cols.append((np.maximum(x - tj, 0) ** 3
                     - np.maximum(x - tkm, 0) ** 3 * (tk - tj) / (tk - tkm)
                     + np.maximum(x - tk, 0) ** 3 * (tkm - tj) / (tk - tkm)) / (tk - t1) ** 2)
    return np.column_stack(cols)


def attach(d, var, tag, knots=None):
    if knots is None:
        knots = np.percentile(d[var].dropna(), KNOT_PCT)
    B = basis(d[var].values, knots)
    names = [f"{tag}{i+1}" for i in range(B.shape[1])]
    for i, n in enumerate(names):
        d[n] = B[:, i]
    return names, knots


def shape(m, terms, raw, knots, pval):
    q = np.percentile(raw.dropna(), Q)
    b = np.array([m.params[t] for t in terms])
    v = (basis(q, knots) - basis(np.array([q[2]]), knots)) @ b
    txt = "  ".join(f"{x:+.2f}" for x in v)
    return v, (f"    fitted at p10..p90: {txt}   span {v.max()-v.min():.2f} C"
               + ("   <- a straight line cannot represent this" if pval < 0.05 else ""))


def report(rows, where, linear, lin_p, curv_p, joint_p, n, note=""):
    rows.append(dict(where=where, linear_slope=linear, linear_p=lin_p,
                     curvature_p=curv_p, spline_joint_p=joint_p, n=n, note=note))


def level(rows):
    d = pd.read_csv(IN + "uhi_level_model_cities.csv")
    rep = pd.read_csv(IN + "representativeness.csv")[["CityID", "country"]].drop_duplicates("CityID")
    d = d.merge(rep, on="CityID", how="left")
    d = d[d["pop"] > 0].dropna(subset=["uhi", "ln_density", "ln_pop", "ln_gdp", "country"]).copy()
    g = pd.factorize(d.country)[0]
    print("\n" + "=" * 78)
    print(f"LEVEL, between cities   {len(d):,} cities, {d.country.nunique()} countries")
    print("=" * 78)
    for tag, ctrl in [("density with size held fixed", "ln_pop"),
                      ("density with size and income", "ln_pop + ln_gdp")]:
        lin = smf.ols(f"uhi ~ ln_density + {ctrl}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        terms, knots = attach(d, "ln_density", "dn")
        sp = smf.ols(f"uhi ~ " + " + ".join(terms) + f" + {ctrl}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        cp = float(sp.f_test([f"{t} = 0" for t in terms[1:]]).pvalue)
        jp = float(sp.f_test([f"{t} = 0" for t in terms]).pvalue)
        print(f"\n  {tag}")
        print(f"    linear  {lin.params['ln_density']:+.3f} (p={lin.pvalues['ln_density']:.4f})"
              f"   R2 {lin.rsquared:.3f}")
        print(f"    spline  joint p={jp:.4f}   curvature p={cp:.4f}   R2 {sp.rsquared:.3f}")
        v, txt = shape(sp, terms, d.ln_density, knots, cp)
        print(txt)
        report(rows, f"level: {tag}", lin.params["ln_density"], lin.pvalues["ln_density"],
               cp, jp, int(lin.nobs))


def trend(rows):
    p = pd.read_csv(IN + "hist_stage3_panel.csv")
    print("\n" + "=" * 78)
    print("TREND, within cities")
    print("=" * 78)
    for tag, sub, ctrl in [("2000-2020, the paper's window", p[p.year >= 2000], ["frac_built"]),
                           ("1975-2020, the full panel", p, ["frac_built"])]:
        s = sub.dropna(subset=["uhi_obs", "ln_popdensity"] + ctrl).drop_duplicates(
            ["CityID", "year"]).copy()
        within = s.ln_popdensity - s.groupby("CityID").ln_popdensity.transform("mean")
        share = within.var() / s.ln_popdensity.var()
        terms, knots = attach(s, "ln_popdensity", "dn")
        si = s.set_index(["CityID", "year"])
        lin = PanelOLS.from_formula("uhi_obs ~ 1 + ln_popdensity + " + " + ".join(ctrl)
                                    + " + EntityEffects + TimeEffects", data=si,
                                    drop_absorbed=True).fit(cov_type="clustered",
                                                            cluster_entity=True)
        sp = PanelOLS.from_formula("uhi_obs ~ 1 + " + " + ".join(terms + ctrl)
                                   + " + EntityEffects + TimeEffects", data=si,
                                   drop_absorbed=True).fit(cov_type="clustered",
                                                           cluster_entity=True)
        live = [t for t in terms[1:] if t in sp.params.index]
        cp = float(sp.wald_test(formula=" = ".join(live) + " = 0").pval) if live else np.nan
        jp = float(sp.wald_test(formula=" = ".join(
            [t for t in terms if t in sp.params.index]) + " = 0").pval)
        print(f"\n  {tag}   {int(lin.nobs):,} city-epochs, "
              f"{si.index.get_level_values(0).nunique():,} cities")
        print(f"    only {100*share:.1f}% of the variation in density is within-city; the rest is"
              " absorbed by the city effects")
        print(f"    linear  {lin.params['ln_popdensity']:+.3f} "
              f"(p={lin.pvalues['ln_popdensity']:.4f})")
        print(f"    spline  joint p={jp:.4f}   curvature p={cp:.4f}"
              + ("   curvature not resolvable on this little movement" if not np.isfinite(cp)
                 or cp > 0.05 else "   <- curvature is real even within cities"))
        report(rows, f"trend: {tag}", lin.params["ln_popdensity"],
               lin.pvalues["ln_popdensity"], cp, jp, int(lin.nobs),
               f"within-city share of density variation {100*share:.1f}%")


def mundlak(rows):
    p = pd.read_csv(IN + "hist_stage3_panel.csv")
    s = p[p.year >= 2000].dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(
        ["CityID", "year"]).copy()
    bar = s.groupby("CityID").ln_popdensity.transform("mean")
    s["dev"] = s.ln_popdensity - bar
    s["bar"] = bar
    print("\n" + "=" * 78)
    print("MUNDLAK, the equality test the paper rests the separation on")
    print("=" * 78)
    si = s.set_index(["CityID", "year"])
    m = PanelOLS.from_formula("uhi_obs ~ 1 + dev + bar + TimeEffects", data=si,
                              drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    t = m.wald_test(formula="dev = bar")
    print(f"\n  as published, both terms linear")
    print(f"    within {m.params['dev']:+.3f} (p={m.pvalues['dev']:.4f})   "
          f"between {m.params['bar']:+.3f} (p={m.pvalues['bar']:.4f})   "
          f"equality p={t.pval:.2e}")
    # the between term is the one with room to bend: city means span the full density range
    terms, knots = attach(s, "bar", "bb")
    si = s.set_index(["CityID", "year"])
    m2 = PanelOLS.from_formula("uhi_obs ~ 1 + dev + " + " + ".join(terms) + " + TimeEffects",
                               data=si, drop_absorbed=True).fit(cov_type="clustered",
                                                                cluster_entity=True)
    live = [t_ for t_ in terms[1:] if t_ in m2.params.index]
    cp = float(m2.wald_test(formula=" = ".join(live) + " = 0").pval) if live else np.nan
    print(f"\n  with the between term free to bend")
    print(f"    within {m2.params['dev']:+.3f} (p={m2.pvalues['dev']:.4f})   "
          f"between: curvature p={cp:.4f}")
    v, txt = shape(m2, [t_ for t_ in terms if t_ in m2.params.index], s.bar, knots, cp)
    print(txt)
    print("\n    The equality test compares one number with one number. If the between-city"
          "\n    relationship is a curve, the test is comparing the within-city slope against"
          "\n    an average of that curve, which is a weaker statement than it appears.")
    report(rows, "mundlak: between term", m.params["bar"], m.pvalues["bar"], cp, np.nan,
           int(m.nobs), f"equality p={t.pval:.2e} when both are linear")


def main():
    rows = []
    level(rows)
    trend(rows)
    mundlak(rows)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT + "density_shape_check.csv", index=False)
    print(f"\nwrote {OUT}density_shape_check.csv")


if __name__ == "__main__":
    main()
