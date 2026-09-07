"""groupings.py — the table behind Supplementary Fig. S1: median UHI by continent, income group,
Köppen zone and country, for the annual mean and the night, with a city-bootstrap interval and the
group's median 2000–2020 trend.

Reads the deposited records only: broad_city_uhi.csv (the 2001–2020 level, clean-rural screened
columns) and city_uhi_epoch_panel.csv (the within-city trend) from the air record, and
city_groupings.csv (continent, income group, Köppen zone) from the companion record.
Writes data/uhi_by_grouping.csv, which make_main_figures.py draws.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from uhi_paths import find
ROOT = os.path.dirname(HERE)

MIN_N, MIN_N_COUNTRY, N_BOOT, SEED = 8, 15, 2000, 0
CONTINENTS = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
INCOME = ["High", "Upper-middle", "Lower-middle", "Low"]
KOPPEN = ["A tropical", "B arid", "C temperate", "D continental", "E polar"]

lvl = pd.read_csv(find("broad_city_uhi.csv"))
grp = pd.read_csv(find("city_groupings.csv")).dropna(subset=["CityID"])
grp["CityID"] = grp.CityID.astype(int)
ep = pd.read_csv(find("city_uhi_epoch_panel.csv"))


def slope_per_decade(g):
    g = g.dropna(subset=["uhi_obs_screened"])
    return np.polyfit(g.year, g.uhi_obs_screened, 1)[0] * 10 if g.year.nunique() >= 3 else np.nan


tr = (ep[ep.year >= 2000].groupby("CityID")[["year", "uhi_obs_screened"]]
      .apply(slope_per_decade).rename("trend").reset_index())
d = (grp[["CityID", "country", "continent", "income", "koppen"]]
     .merge(lvl[["CityID", "uhi_tavg_screened", "uhi_tmin_screened"]], on="CityID", how="inner")
     .merge(tr, on="CityID", how="left")
     .rename(columns={"uhi_tavg_screened": "uhi", "uhi_tmin_screened": "uhi_tmin"})
     .dropna(subset=["uhi"]))
rng = np.random.default_rng(SEED)


def boot_median(v, floor=MIN_N):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < floor:
        return np.nan, np.nan, np.nan, len(v)
    bs = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)


majors = [c for c in d.country.value_counts().index if (d.country == c).sum() >= MIN_N_COUNTRY][:12]
rows = []
for col, order, floor in [("continent", CONTINENTS, MIN_N), ("income", INCOME, MIN_N),
                          ("koppen", KOPPEN, MIN_N), ("country", majors, MIN_N_COUNTRY)]:
    for g in order:
        s = d[d[col] == g]
        if len(s) < floor:
            continue
        m, lo, hi, n = boot_median(s.uhi); mn, lon, hin, nn = boot_median(s.uhi_tmin)
        rows.append(dict(split=col, group=g, n=n, mean_uhi=round(m, 4), mean_lo=round(lo, 4), mean_hi=round(hi, 4),
                         n_night=nn, night_uhi=round(mn, 4) if np.isfinite(mn) else np.nan,
                         night_lo=round(lon, 4) if np.isfinite(lon) else np.nan,
                         night_hi=round(hin, 4) if np.isfinite(hin) else np.nan,
                         trend=round(float(s.trend.median()), 4), pct_negative=round(100 * float((s.uhi < 0).mean()))))
out = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
out.to_csv(os.path.join(ROOT, "data", "uhi_by_grouping.csv"), index=False)
print(f"{len(d)} cities; {len(out)} groups shown, night estimate missing for {int(out.night_uhi.isna().sum())}")
print(out.to_string(index=False))
