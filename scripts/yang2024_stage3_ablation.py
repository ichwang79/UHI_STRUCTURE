"""The Yang et al. (2024) cross-instrument check of Section 5.1: does an independent canopy
air-temperature UHI product, built by other hands from a different satellite record, recover a
within-city density response under the same specification used for the station panel?

Yang2024_UHII_dataset (Yang et al. 2024, Remote Sensing of Environment) is a global canopy
air-temperature UHI intensity dataset derived from ERA5-Land and satellite land-surface
temperature, keyed on the dataset's own UrbanId rather than this paper's CityID. `yang2024_
regression_panel.csv` is the pre-built join: Yang2024's own day/night/mean UHI values on the left,
this paper's own ln_popdensity, the GDP restricted-cubic-spline basis and built-up fraction (from
GHS-UCDB, matched by CityID) on the right. The UHI values and the predictors therefore come from
two different instruments and two different processing chains, joined only on which city each
observation is.

Four specifications are fitted for each of the three UHI elements, as an ablation: the full
specification (density, the GDP spline, built-up fraction), each control dropped in turn, and
density alone ("Minus both"), which is the specification comparable to the paper's own
`si_robustness_suite.py` density-only rows. All are entity+year fixed-effects panel regressions,
clustered on UrbanId.

Output: data/results/yang2024_stage3_ablation_full_regressions.csv
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, warnings
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
# the panel ships with the companion data deposit; the wide all-indicator table does not, so the
# per-variant sweep is skipped when it is absent (see README, "What this cannot rebuild")
COMPANION = Path(os.environ.get("UHI_AIR_COMPANION", ROOT.parent / "companion_release"))
PANEL = COMPANION / "yang2024_regression_panel.csv"
# Yang et al. release several UHII variants side by side, one per satellite/overpass combination
# (AMod2/Mod1/Mod2 = Terra, day and night overpasses two ways; Myd1/Myd2 = Aqua; SMod2/SMyd1 =
# smoothed Terra/Aqua). `sat_day/sat_night` in the regression panel is one of these, not an
# average across them, so the density-only specification is also run on every other variant, to
# show whether the within-city response is a property of the product Yang et al. lead with or one
# specific to it.
WIDE = Path(os.environ.get("YANG_WIDE", COMPANION / "yang2024_all_indicators_wide.csv"))

SPECS = {
    "Full (density+RCS(GDP)+built)": ["ln_popdensity", "gdp_rcs1", "gdp_rcs2", "gdp_rcs3", "frac_urban_built"],
    "Minus built_frac": ["ln_popdensity", "gdp_rcs1", "gdp_rcs2", "gdp_rcs3"],
    "Minus RCS(GDP)": ["ln_popdensity", "frac_urban_built"],
    "Minus both": ["ln_popdensity"],
}


def fit(d, dv, cols):
    # Indexed on CityID, not UrbanId: a handful of Yang2024 UrbanIds are shared by more than one
    # of this paper's cities (350 of 24,520 rows), and each keeps its own predictor values, so the
    # panel's entity is this paper's city rather than Yang2024's urban cluster.
    s = d.dropna(subset=[dv, *cols]).drop_duplicates(["CityID", "year"]).set_index(["CityID", "year"])
    m = PanelOLS(s[dv], s[cols], entity_effects=True, time_effects=True, drop_absorbed=True)
    f = m.fit(cov_type="clustered", cluster_entity=True)
    row = {"DV": dv, "Spec": [k for k, v in SPECS.items() if v == cols][0], "n_obs": int(f.nobs),
           "R2within": f.rsquared_within}
    for c in ["ln_popdensity", "gdp_rcs1", "gdp_rcs2", "gdp_rcs3", "frac_urban_built"]:
        if c in cols:
            row[f"{c}_coef"] = f.params[c]
            row[f"{c}_p"] = f.pvalues[c]
    return row


def main():
    d = pd.read_csv(PANEL)
    rows = []
    for dv, lab in (("sat_day", "SAT Day"), ("sat_night", "SAT Night"), ("sat_mean", "SAT Mean")):
        for spec, cols in SPECS.items():
            r = fit(d, dv, cols)
            r["DV"] = lab
            rows.append(r)
            print(f"  {lab:10} {spec:32} density {r.get('ln_popdensity_coef', float('nan')):+.4f}"
                  f" (p {r.get('ln_popdensity_p', float('nan')):.2e})  n={r['n_obs']}")
    out = pd.DataFrame(rows)
    dest = ROOT / "data" / "results" / "yang2024_stage3_ablation_full_regressions.csv"
    out.to_csv(dest, index=False)
    print(f"\nsaved {len(out)} rows -> {dest}")

    print("\n=== density-only, every Yang et al. satellite/overpass variant ===")
    xwalk = d[["CityID", "UrbanId", "year", "ln_popdensity"]].drop_duplicates()
    wide = pd.read_csv(WIDE)
    merged = xwalk.merge(wide, on=["UrbanId", "year"], how="inner")
    variant_rows = []
    indicators = ["sat_day", "sat_night"] + [c for c in wide.columns if c not in ("UrbanId", "year")]
    base = d[["CityID", "UrbanId", "year", "ln_popdensity", "sat_day", "sat_night"]]
    full = base.merge(wide, on=["UrbanId", "year"], how="left")
    for ind in indicators:
        r = fit(full, ind, ["ln_popdensity"])
        r["indicator"] = ind
        variant_rows.append(r)
        print(f"  {ind:12} density {r['ln_popdensity_coef']:+.4f} (p {r['ln_popdensity_p']:.2e})"
              f"  n={r['n_obs']}")
    vout = pd.DataFrame(variant_rows)[["indicator", "n_obs", "ln_popdensity_coef", "ln_popdensity_p"]]
    vout.columns = ["indicator", "n_obs", "density_coef", "density_p"]
    vdest = ROOT / "data" / "results" / "yang2024_all_indicators_density_fe_results.csv"
    vout.to_csv(vdest, index=False)
    n_positive = (vout.density_coef > 0).sum()
    n_sig = ((vout.density_coef > 0) & (vout.density_p < 0.05)).sum()
    print(f"\npositive in {n_positive} of {len(vout)} variants, significant and positive in {n_sig}")
    print(f"saved -> {vdest}")


if __name__ == "__main__":
    main()
