"""The Supplementary Information's robustness suite for the within-city density trend.

Every check here is run on the anomaly-referenced epoch panel (`hist_stage3_panel.csv`), the
construction adopted in Section 4.1. The earlier varying-reference panel is kept alongside it as
`hist_stage3_panel_varying.csv`, and any number carried over from before that correction is wrong
for the trend: this script exists so that the SI quotes nothing that predates it.

The estimator is the corrected Mundlak form of Section 4.3. Year means are removed first, the
regressor is then split into a city mean and a deviation from it, and both enter one OLS with
standard errors clustered on the city. `within` is therefore the two-way fixed-effects estimate,
`between` the cross-city one, and `eq_p` the test that one coefficient describes both.

Output: data/results/si_robustness_suite.csv, one row per check, plus a printed log.
"""
from __future__ import annotations
import os, warnings, math
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
# P3 is the air-temperature record (city_station_match_broad.csv lives there); GROUPINGS is the
# folder holding city_groupings.csv (continent/income/Köppen lookup), which the companion record ships.
P3 = Path(os.environ.get("UHI_AIR_DATA", ROOT / "data" / "air_record"))
GROUPINGS = Path(os.environ.get("UHI_AIR_GROUPINGS", os.environ.get("UHI_AIR_COMPANION", ROOT / "data" / "companion")))
RNG = np.random.default_rng(0)
MIN_CITIES = 50   # a subgroup below this is reported as too small rather than estimated
EP = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# continents absent from broad_groupings_cities.csv, filled by hand
EXTRA_CONT = {"ARG": "South America", "URY": "South America", "MDA": "Europe",
              "SVK": "Europe", "TJK": "Asia", "TKM": "Asia", "ZAF": "Africa"}


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/inputs/hist_stage3_panel.csv")
    m = pd.read_csv(P3 / "city_station_match_broad.csv")[["city_id", "country", "urban_km"]]
    _gf = next((GROUPINGS / f for f in ("city_groupings.csv", "broad_groupings_cities.csv")
            if (GROUPINGS / f).exists()), GROUPINGS / "city_groupings.csv")
    g = pd.read_csv(_gf)[["city_id", "continent", "income",
                                                        "koppen_main_group"]]
    d = d.merge(m, left_on="CityID", right_on="city_id", how="left")
    d = d.merge(g.drop_duplicates("city_id"), on="city_id", how="left")
    d["continent"] = d["continent"].fillna(d["country"].map(EXTRA_CONT))
    return d


def mundlak(d, x="ln_popdensity", extra=(), cluster="CityID"):
    """Corrected Mundlak: year means removed, then the city mean and deviation formed."""
    cols = ["uhi_obs", x, *extra]
    s = d.dropna(subset=cols + [cluster]).drop_duplicates(["CityID", "year"]).copy()
    if s.CityID.nunique() < MIN_CITIES:
        return None
    for v in cols:
        s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby("CityID")["t_" + x].transform("mean")
    s["b"], s["w"] = bar, s["t_" + x] - bar
    rhs = " + ".join(["w", "b"] + ["t_" + v for v in extra])
    f = smf.ols(f"t_uhi_obs ~ {rhs}", data=s).fit(cov_type="cluster",
                                                  cov_kwds={"groups": s[cluster]})
    return dict(within=f.params["w"], within_p=f.pvalues["w"], within_se=f.bse["w"],
                between=f.params["b"], between_p=f.pvalues["b"],
                eq_p=float(f.f_test("w - b = 0").pvalue),
                n_obs=len(s), n_cities=s.CityID.nunique())


def twoway(d, cols=("ln_popdensity",), cov="clustered", **kw):
    s = d.dropna(subset=["uhi_obs", *cols]).drop_duplicates(["CityID", "year"])
    s = s.set_index(["CityID", "year"])
    m = PanelOLS(s["uhi_obs"], s[list(cols)], entity_effects=True, time_effects=True,
                 drop_absorbed=True)
    f = m.fit(cov_type=cov, cluster_entity=(cov == "clustered"), **kw)
    return dict(within=f.params["ln_popdensity"], within_se=f.std_errors["ln_popdensity"],
                within_p=f.pvalues["ln_popdensity"], n_obs=int(f.nobs),
                n_cities=s.index.get_level_values(0).nunique())


ROWS = []


def rec(group, check, r, note=""):
    if r is None:
        print(f"  {check:52} sample too small")
        return
    ROWS.append(dict(group=group, check=check, note=note, **r))
    b = r.get("between")
    tail = "" if b is None else f"  between {b:+.3f}  eq p {r['eq_p']:.2e}"
    print(f"  {check:52} within {r['within']:+.3f} (p {r['within_p']:.1e}){tail}"
          f"  {r['n_cities']} cities")


def main():
    d = load()
    m1 = d  # 1975-2020, the full record
    m2 = d[d.year >= 2000]

    print("=== baselines")
    rec("baseline", "Model 1, 1975-2020, Mundlak", mundlak(m1))
    rec("baseline", "Model 2, 2000-2020, Mundlak", mundlak(m2))
    # the two-way fixed-effects comparison of Section 4.1 carries built-up alongside density
    for y0 in (1975, 1990, 2000):
        rec("baseline", f"two-way FE, {y0}-2020, density + built",
            twoway(d[d.year >= y0], cols=("ln_popdensity", "frac_built")))
        rec("baseline", f"two-way FE, {y0}-2020, density only", twoway(d[d.year >= y0]))

    print("\n=== start year")
    for y0 in (1975, 1980, 1985, 1990, 1995, 2000, 2005):
        rec("window", f"start year {y0}", mundlak(d[d.year >= y0]))

    print("\n=== specification")
    rec("spec", "density only", mundlak(m1))
    rec("spec", "+ built-up fraction", mundlak(m1, extra=["frac_built"]))
    rec("spec", "+ built-up, 2000-2020", mundlak(m2, extra=["frac_built"]))
    rec("spec", "+ income spline, 2000-2020", mundlak(m2, extra=["g1", "g2", "g3"]))
    rec("spec", "+ income spline + built, 2000-2020",
        mundlak(m2, extra=["g1", "g2", "g3", "frac_built"]))
    # pooled and between-only estimators, for the identification contrast
    s = m1.dropna(subset=["uhi_obs", "ln_popdensity"]).copy()
    pooled = smf.ols("uhi_obs ~ ln_popdensity", data=s).fit(cov_type="cluster",
                                                            cov_kwds={"groups": s.CityID})
    ROWS.append(dict(group="spec", check="pooled OLS, no decomposition", note="",
                     within=pooled.params["ln_popdensity"], within_se=pooled.bse["ln_popdensity"],
                     within_p=pooled.pvalues["ln_popdensity"], n_obs=len(s),
                     n_cities=s.CityID.nunique()))
    print(f"  {'pooled OLS, no decomposition':52} slope {pooled.params['ln_popdensity']:+.3f}"
          f" (p {pooled.pvalues['ln_popdensity']:.1e})")

    print("\n=== sample: region")
    rec("sample", "excluding the United States", mundlak(m1[m1.country != "USA"]))
    rec("sample", "excluding the United States, 2000-2020", mundlak(m2[m2.country != "USA"]))
    for c in sorted(m1.continent.dropna().unique()):
        sub = m1[m1.continent != c]
        rec("sample", f"excluding {c}", mundlak(sub))
    rec("sample", "excluding North America, 2000-2020",
        mundlak(m2[m2.continent != "North America"]))
    big = (m1.groupby("country").CityID.nunique().sort_values(ascending=False))
    for c in big[big >= 20].index:
        rec("sample", f"excluding {c}", mundlak(m1[m1.country != c]))

    print("\n=== sample: is the North American dependence a power problem?")
    # A shrinking subsample can lose significance for two reasons: the estimate moves, or the
    # interval widens. These separate them, because the answer decides how the limit is reported.
    # Run on both windows: the paper's primary window is 2000-2020, so that is the version quoted in the main
    # text, with the full-record (m1) version carried alongside as a check that the collapse is not
    # itself a primary-window artifact.
    for window, base in (("2000-2020", m2), ("1975-2020", m1)):
        ss = base.dropna(subset=["uhi_obs", "ln_popdensity", "continent"]).copy()
        ss = ss.drop_duplicates(["CityID", "year"])
        for v in ("uhi_obs", "ln_popdensity"):
            ss["t_" + v] = ss[v] - ss.groupby("year")[v].transform("mean")
        bar = ss.groupby("CityID").t_ln_popdensity.transform("mean")
        ss["b"], ss["w"] = bar, ss.t_ln_popdensity - bar
        ss["na"] = (ss.continent == "North America").astype(float)
        ss["w_na"], ss["b_na"] = ss.w * ss.na, ss.b * ss.na
        f = smf.ols("t_uhi_obs ~ w + b + w_na + b_na + na", data=ss).fit(
            cov_type="cluster", cov_kwds={"groups": ss.CityID})
        for tag, k in (("rest of world", "w"), ("North American excess", "w_na")):
            ROWS.append(dict(group="region", check=f"interaction, within: {tag}, {window}",
                             note="one model", within=f.params[k], within_se=f.bse[k],
                             within_p=f.pvalues[k], n_obs=len(ss), n_cities=ss.CityID.nunique()))
            print(f"  {'interaction, within: ' + tag + f' ({window})':52}"
                  f" {f.params[k]:+.3f} (p {f.pvalues[k]:.1e})")
    # and whether simply matching the amount of densification closes the gap
    rng = m1.groupby("CityID").ln_popdensity.agg(lambda v: v.max() - v.min())
    cut = rng[m1[m1.continent == "North America"].CityID.unique()].median()
    keep = rng[rng >= cut].index
    for lab, sub in (("North America", m1[m1.continent == "North America"]),
                     ("rest of world", m1[m1.continent != "North America"])):
        rec("region", f"{lab}, all cities", mundlak(sub))
        rec("region", f"{lab}, densified at least the NA median",
            mundlak(sub[sub.CityID.isin(keep)]), note=f"threshold {cut:.3f}")

    print("\n=== sample: thresholds")
    ne = m1.groupby("CityID").year.nunique()
    for k in (3, 5, 7):
        rec("sample", f"cities with at least {k} epochs",
            mundlak(m1[m1.CityID.isin(ne[ne >= k].index)]))
    rec("sample", "balanced panel, all 10 epochs",
        mundlak(m1[m1.CityID.isin(ne[ne == 10].index)]))
    for k in (3, 10, 20):
        rec("sample", f"at least {k} rural references", mundlak(m1[m1.n_ref >= k]))
    rec("sample", "urban station within 10 km of centroid", mundlak(m1[m1.urban_km <= 10]))

    print("\n=== sample: trimming")
    cm = m1.groupby("CityID").uhi_obs.mean()
    lo, hi = cm.quantile([0.01, 0.99])
    rec("sample", "drop the 1 % most extreme city UHI levels",
        mundlak(m1[m1.CityID.isin(cm[(cm > lo) & (cm < hi)].index)]))
    rng = m1.groupby("CityID").ln_popdensity.agg(lambda v: v.max() - v.min())
    rec("sample", "drop the 5 % largest density changes",
        mundlak(m1[m1.CityID.isin(rng[rng <= rng.quantile(0.95)].index)]))

    print("\n=== subgroups")
    for z, lab in [("A", "tropical"), ("B", "arid"), ("C", "temperate"), ("D", "continental")]:
        rec("subgroup", f"Koppen {z} ({lab})", mundlak(m1[m1.koppen_main_group == z]))
    for g in ("High", "Upper-middle", "Lower-middle"):
        rec("subgroup", f"income group {g}", mundlak(m1[m1.income == g]))

    # Development phase. The urbanising group is small, so the interval comes from a city bootstrap
    # rather than from the clustered standard error alone.
    ph = m1.dropna(subset=["income"]).copy()
    ph["phase"] = np.where(ph.income == "High", "mature", "urbanising")
    for lab in ("mature", "urbanising"):
        sub = ph[ph.phase == lab]
        global MIN_CITIES
        keep, MIN_CITIES = MIN_CITIES, 30
        r = mundlak(sub)
        MIN_CITIES = keep
        s = sub.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).copy()
        for v in ("uhi_obs", "ln_popdensity"):
            s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
        bar = s.groupby("CityID").t_ln_popdensity.transform("mean")
        s["b"], s["w"] = bar, s.t_ln_popdensity - bar
        keys = s.CityID.unique(); draws = []
        for _ in range(600):
            take = RNG.choice(keys, size=len(keys), replace=True)
            dd = pd.concat([s[s.CityID == k] for k in take], ignore_index=True)
            try:
                draws.append(smf.ols("t_uhi_obs ~ w + b", data=dd).fit().params["w"])
            except Exception:
                pass
        arr = np.array(draws)
        r = dict(r or {}, ci_lo=float(np.percentile(arr, 2.5)), ci_hi=float(np.percentile(arr, 97.5)),
                 frac_positive=float((arr > 0).mean()))
        rec("subgroup", f"development phase: {lab}", r, note=f"{len(arr)} bootstrap draws")

    print("\n=== inference")
    rec("inference", "clustered on city", mundlak(m1))
    rec("inference", "clustered on country", mundlak(m1, cluster="country"))
    rec("inference", "clustered on city, 2000-2020", mundlak(m2))
    rec("inference", "clustered on country, 2000-2020", mundlak(m2, cluster="country"))
    rec("inference", "Driscoll-Kraay spatial HAC",
        twoway(m1, cov="kernel", kernel="bartlett"))

    def boot(base, unit, n=1000):
        s = base.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).copy()
        for v in ("uhi_obs", "ln_popdensity"):
            s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
        bar = s.groupby("CityID").t_ln_popdensity.transform("mean")
        s["b"], s["w"] = bar, s.t_ln_popdensity - bar
        keys = s[unit].dropna().unique()
        out = []
        for _ in range(n):
            take = RNG.choice(keys, size=len(keys), replace=True)
            dd = pd.concat([s[s[unit] == k] for k in take], ignore_index=True)
            try:
                out.append(smf.ols("t_uhi_obs ~ w + b", data=dd).fit().params["w"])
            except Exception:
                pass
        return np.array(out), s

    for window, base in (("1975-2020", m1), ("2000-2020", m2)):
        for unit, lab in (("CityID", "city"), ("country", "country")):
            a, s = boot(base, unit, 1000 if unit == "country" else 400)
            ROWS.append(dict(group="inference", check=f"{lab}-cluster bootstrap, {window}",
                             note=f"{len(a)} draws", within=float(a.mean()), within_se=float(a.std()),
                             within_p=np.nan, ci_lo=float(np.percentile(a, 2.5)),
                             ci_hi=float(np.percentile(a, 97.5)),
                             frac_positive=float((a > 0).mean()), n_obs=len(s),
                             n_cities=s.CityID.nunique()))
            print(f"  {lab+'-cluster bootstrap ('+window+')':52} 95% CI [{np.percentile(a,2.5):+.3f},"
                  f" {np.percentile(a,97.5):+.3f}]  positive in {100*(a>0).mean():.1f}%")

    s = m1.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).copy()
    for v in ("uhi_obs", "ln_popdensity"):
        s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby("CityID").t_ln_popdensity.transform("mean")
    s["b"], s["w"] = bar, s.t_ln_popdensity - bar

    # Pesaran cross-sectional dependence on the two-way FE residuals
    ss = s.set_index(["CityID", "year"])
    res = PanelOLS(ss["uhi_obs"], ss[["ln_popdensity"]], entity_effects=True,
                   time_effects=True).fit().resids.reset_index()
    w = res.pivot_table(index="year", columns="CityID", values="residual")
    C = w.corr(min_periods=5).values
    iu = np.triu_indices_from(C, 1)
    rho = np.nanmean(C[iu])
    ROWS.append(dict(group="inference", check="Pesaran cross-sectional dependence",
                     note="mean pairwise residual correlation", within=float(rho),
                     within_se=np.nan, within_p=np.nan, n_obs=len(s),
                     n_cities=s.CityID.nunique()))
    print(f"  {'Pesaran cross-sectional dependence':52} mean pairwise rho {rho:+.3f}")

    print("\n=== placebo")

    def placebo(base, s0, n=300):
        obs0 = mundlak(base)["within"]
        draws = []
        for _ in range(n):
            dd = s0.copy()
            dd["t_ln_popdensity"] = dd.groupby("CityID").t_ln_popdensity.transform(
                lambda v: RNG.permutation(v.values))
            bar = dd.groupby("CityID").t_ln_popdensity.transform("mean")
            dd["b"], dd["w"] = bar, dd.t_ln_popdensity - bar
            draws.append(smf.ols("t_uhi_obs ~ w + b", data=dd).fit().params["w"])
        return np.array(draws), obs0

    s2 = m2.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).copy()
    for v in ("uhi_obs", "ln_popdensity"):
        s2["t_" + v] = s2[v] - s2.groupby("year")[v].transform("mean")
    bar2 = s2.groupby("CityID").t_ln_popdensity.transform("mean")
    s2["b"], s2["w"] = bar2, s2.t_ln_popdensity - bar2

    for window, base, sx in (("1975-2020", m1, s), ("2000-2020", m2, s2)):
        a, obs = placebo(base, sx)
        ROWS.append(dict(group="placebo", check=f"within-city permutation of density, {window}",
                         note="300 draws", within=float(a.mean()), within_se=float(a.std()),
                         within_p=float((np.abs(a) >= abs(obs)).mean()),
                         ci_lo=float(np.percentile(a, 2.5)), ci_hi=float(np.percentile(a, 97.5)),
                         n_obs=len(sx), n_cities=sx.CityID.nunique()))
        print(f"  {'within-city permutation of density (' + window + ')':52} {a.mean():+.4f}"
              f" +/- {a.std():.4f}  max |draw| {np.abs(a).max():.3f} against observed {obs:+.3f}")

    print("\n=== day and night channels")
    dn = pd.read_csv(ROOT / "data/inputs/hist_stage3_panel_daynight.csv")
    dn = dn.merge(pd.read_csv(P3 / "city_station_match_broad.csv")[["city_id", "country"]],
                  left_on="CityID", right_on="city_id", how="left")
    for ch, col in (("mean", "uhi_mean"), ("night", "uhi_night"), ("day", "uhi_day")):
        sub = dn.rename(columns={col: "uhi_obs"})
        rec("daynight", f"{ch} channel, 1975-2020", mundlak(sub))
        rec("daynight", f"{ch} channel, 2000-2020", mundlak(sub[sub.year >= 2000]))

    out = pd.DataFrame(ROWS)
    dest = ROOT / "data/results/si_robustness_suite.csv"
    out.to_csv(dest, index=False)
    print(f"\nsaved {len(out)} rows -> {dest}")


if __name__ == "__main__":
    main()
