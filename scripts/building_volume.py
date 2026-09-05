"""Does building volume (GHS-BUILT-V, fixed GHS-UCDB polygon) absorb the within-city population
term on the station UHI panel? 2000-2020 primary window; feeds Table 1 and Supplementary Table S6."""
import os, warnings, glob
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from supplement_paths import AIR, COMP, CODE_IN, EXTRA as XTRA
warnings.filterwarnings("ignore")
P = pd.read_csv(CODE_IN + "within_city_panel.csv")
DN = pd.read_csv(AIR + "city_uhi_epoch_panel_daynight.csv")
G = pd.read_csv(COMP + "city_groupings.csv")
V = pd.read_csv(XTRA + "gee_built_volume_extraction.csv")
print("volume match_type", V.match_type.value_counts().to_dict(), "truncated", V.possibly_truncated.sum())
V = V[V.volume_density_m > 0].copy(); V["ln_vol"] = np.log(V.volume_density_m)
V = V[["CityID", "year", "ln_vol", "possibly_truncated"]]

# continent: groupings first, then any CityID->ISO table found
G = G.dropna(subset=["CityID"]).drop_duplicates("CityID"); G["CityID"] = G.CityID.astype(int)
cont = G.set_index("CityID")["continent"].to_dict()
kop = G.set_index("CityID")["koppen_main_group"].to_dict()
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iso_cont.py")).read())
BASE = {"USA":"North America","CAN":"North America","MEX":"North America","JPN":"Asia","KOR":"Asia","CHN":"Asia","TWN":"Asia",
        "AUS":"Oceania","NZL":"Oceania","BRA":"South America","CHL":"South America","RUS":"Europe","TUR":"Asia","IDN":"Asia","MYS":"Asia"}
EU = "AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE GBR NOR CHE ISL SRB BIH MKD ALB MNE UKR BLR MDA XKX".split()
ISO = {**{k: "Europe" for k in EU}, **BASE, **ISO_EXTRA}
# continent comes from city_groupings.csv; ISO lookup kept for any city the groupings file lacks


M = P.merge(V, on=["CityID", "year"], how="left")
M["continent"] = M.CityID.map(cont); M["koppen"] = M.CityID.map(kop)
H = M[(M.year >= 2000) & M.ln_vol.notna()].copy()
print(f"primary window with volume: {len(H)} rows, {H.CityID.nunique()} cities; continent known for {H.continent.notna().mean():.0%}")
print("continents:", H.groupby('continent').CityID.nunique().to_dict())

def fe(d, rhs, y="uhi_obs"):
    d = d.dropna(subset=[y] + rhs.split(" + ")).copy()
    f = smf.ols(f"{y} ~ {rhs} + C(CityID) + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.CityID})
    return " | ".join(f"{k} {f.params[k]:+.3f} (p {f.pvalues[k]:.3f})" for k in rhs.split(" + ")) + f"  n={d.CityID.nunique()}"

def mundlak(d, keys, y="uhi_obs", extra=()):
    cols = [y, *keys, *extra]
    s = d.dropna(subset=cols).drop_duplicates(["CityID", "year"]).copy()
    for v in cols:
        s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    terms = []
    for k in keys:
        bar = s.groupby("CityID")["t_" + k].transform("mean")
        s["b_" + k], s["w_" + k] = bar, s["t_" + k] - bar; terms += ["w_" + k, "b_" + k]
    f = smf.ols(f"t_{y} ~ " + " + ".join(terms + ["t_" + v for v in extra]), data=s).fit(cov_type="cluster", cov_kwds={"groups": s.CityID})
    out = " | ".join(f"{t} {f.params[t]:+.3f} (p {f.pvalues[t]:.3f})" for t in terms)
    eq = " ; ".join(f"eq[{k}] p={float(f.f_test(f'w_{k} - b_{k} = 0').pvalue):.3g}" for k in keys)
    return f"{out}  {eq}  n={s.CityID.nunique()}"

print("\n== A. Two-way FE, 2000-2020, same 1,106-city sample")
for rhs in ["ln_popdensity", "ln_vol", "ln_popdensity + ln_vol", "ln_popdensity + ln_vol + frac_built"]:
    print(f"  {rhs:40s} {fe(H, rhs)}")
print("  (drop possibly_truncated)", fe(H[~H.possibly_truncated.astype(bool)], "ln_popdensity + ln_vol"))
w = H.copy()
for v in ["ln_popdensity", "ln_vol"]:
    w[v + "_w"] = w[v] - w.groupby("CityID")[v].transform("mean")
print(f"  within-city corr(pop, vol) = {w[['ln_popdensity_w','ln_vol_w']].corr().iloc[0,1]:.3f}; within SD pop {w.ln_popdensity_w.std():.3f}, vol {w.ln_vol_w.std():.3f}")

print("\n== B. Mundlak (year-demeaned), 2000-2020")
print("  pop only     ", mundlak(H, ["ln_popdensity"]))
print("  vol only     ", mundlak(H, ["ln_vol"]))
print("  pop + vol    ", mundlak(H, ["ln_popdensity", "ln_vol"]))
print("  + built      ", mundlak(H, ["ln_popdensity", "ln_vol"], extra=["frac_built"]))

print("\n== C. By element (day/night panel merged with predictors)")
D = DN[["CityID", "year", "uhi_mean", "uhi_night", "uhi_day"]].merge(P[["CityID", "year", "ln_popdensity", "frac_built"]], on=["CityID", "year"]).merge(V, on=["CityID", "year"])
D = D[D.year >= 2000]
for y in ["uhi_night", "uhi_day", "uhi_mean"]:
    for rhs in ["ln_popdensity", "ln_vol", "ln_popdensity + ln_vol"]:
        print(f"  {y:10s} {rhs:26s} {fe(D, rhs, y)}")
    print(f"  {y:10s} Mundlak pop+vol  {mundlak(D, ['ln_popdensity','ln_vol'], y)}")

print("\n== D. By region, 2000-2020 (FE, pop + vol)")
for reg in ["North America", "Europe", "Asia"]:
    s = H[H.continent == reg]
    if s.CityID.nunique() < 30: continue
    print(f"  {reg:14s} pop alone: {fe(s, 'ln_popdensity')}")
    print(f"  {'':14s} vol alone: {fe(s, 'ln_vol')}")
    print(f"  {'':14s} both     : {fe(s, 'ln_popdensity + ln_vol')}")
s = H[H.continent.notna() & (H.continent != "North America")]
print(f"  {'ex-NA':14s} both     : {fe(s, 'ln_popdensity + ln_vol')}")
print(f"  {'ex-NA':14s} vol alone: {fe(s, 'ln_vol')}")
for reg in ["North America", "Europe"]:
    s = w[w.continent == reg]
    print(f"  within SD {reg}: pop {s.ln_popdensity_w.std():.3f} vol {s.ln_vol_w.std():.3f}; median growth factor vol {np.exp(s.groupby('CityID').ln_vol.agg(lambda x: x.max()-x.min()).median()):.3f}")

print("\n== D2. Night by region")
D["continent"] = D.CityID.map(cont)
for reg in ["North America", "Europe"]:
    s = D[D.continent == reg]
    print(f"  {reg:14s} night both: {fe(s, 'ln_popdensity + ln_vol', 'uhi_night')}")
    print(f"  {'':14s} day   both: {fe(s, 'ln_popdensity + ln_vol', 'uhi_day')}")

print("\n== E. By Köppen main group (FE, pop + vol)")
for k in ["A", "B", "C", "D"]:
    s = H[H.koppen == k]
    if s.CityID.nunique() < 30: continue
    print(f"  {k} ({s.CityID.nunique()}): pop alone {fe(s, 'ln_popdensity')}")
    print(f"       both: {fe(s, 'ln_popdensity + ln_vol')}")
