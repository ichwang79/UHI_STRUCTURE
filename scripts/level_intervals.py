"""95 % intervals for the level quantities (Supplementary Table S3: zone slopes and means, station-distance
bins), the income split, the Mundlak terms by element (Table 1), the moving-boundary decomposition
(Supplementary Table S5) and the seasonal level means of S1.7. Levels from the code deposit's data/inputs;
panel objects from building_volume.py."""
import os, sys, glob, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from supplement_paths import CODE_IN, EXTRA
IN = CODE_IN
def ci(est, se): return f"{est:+.3f} (se {se:.3f}, 95% CI {est-1.96*se:+.2f} to {est+1.96*se:+.2f})"
def mci(x): x = pd.Series(x).dropna(); se = x.std()/np.sqrt(len(x)); return f"mean {ci(x.mean(), se)} n={len(x)}"

pr = pd.read_csv(IN + "city_uhi_predictors.csv"); pr["uhi_max"] = 2*pr.uhi_tavg - pr.uhi_tmin
rep = pd.read_csv(IN + "representativeness.csv")[["CityID", "pop", "abslat", "income", "kop", "koppen_main_group"]]
mt = pd.read_csv(IN + "city_station_match.csv")[["city_id", "urban_km", "n_rural"]]
d = pr.merge(rep, on="CityID", how="left").merge(mt, on="city_id", how="left")
d = d.dropna(subset=["ln_popdensity"]); d = d[d["pop"] > 0].copy(); d["lp"] = np.log10(d["pop"])
zone = d.koppen_main_group.fillna(d.kop).astype(str).str[0]
d["zone"] = zone.map({"A": "tropical", "B": "arid", "C": "temperate", "D": "continental", "E": "polar"})
def cl(df, f):
    return smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.country)[0]}) if df.country.nunique() > 1 else smf.ols(f, data=df).fit(cov_type="HC1")
s = d.dropna(subset=["uhi_tmin", "lp"]); print("== reference check night", ci(cl(s, "uhi_tmin ~ lp").params["lp"], cl(s, "uhi_tmin ~ lp").bse["lp"]), "n", len(s))
print("== size law by Köppen zone, night")
for z in ["tropical", "arid", "temperate", "continental"]:
    ss = s[s.zone == z]; m = cl(ss, "uhi_tmin ~ lp"); print(f"  {z:12s} {ci(m.params['lp'], m.bse['lp'])} n={len(ss)}  | night mean {mci(ss.uhi_tmin)}")
print("  all non-tropical night", mci(s[s.zone != "tropical"].uhi_tmin))
m = cl(s.dropna(subset=["zone"]), "uhi_tmin ~ lp * C(zone)"); print("  size x zone joint p =", float(m.f_test(" = ".join([f"lp:C(zone)[T.{z}]" for z in ["continental", "temperate", "tropical"]]) + " = 0").pvalue) if False else "see below")
mm = smf.ols("uhi_tmin ~ lp * C(zone)", data=s.dropna(subset=["zone"])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.dropna(subset=["zone"]).country)[0]})
names = [n for n in mm.params.index if n.startswith("lp:")]; print("  size x zone joint p =", float(mm.f_test(", ".join(f"{n} = 0" for n in names)).pvalue))
print("== night level by urban-station distance bin")
for lab, ss in [("<=5 km", s[s.urban_km <= 5]), ("5-15 km", s[(s.urban_km > 5) & (s.urban_km <= 15)]), (">15 km", s[s.urban_km > 15])]:
    print(f"  {lab:8s} {mci(ss.uhi_tmin)}")
ss = s.dropna(subset=["urban_km"]); ss = ss.assign(dist=ss.urban_km); m = cl(ss, "uhi_tmin ~ lp * dist"); print("  size x distance", ci(m.params["lp:dist"], m.bse["lp:dist"]))
m = cl(d.dropna(subset=["uhi_tavg", "urban_km"]), "uhi_tavg ~ urban_km"); print("  mean-UHI slope on distance", ci(m.params["urban_km"], m.bse["urban_km"]), "p", round(m.pvalues["urban_km"], 3))
print("== income split, night and mean")
for inc in ["High", "Upper-middle"]:
    for y in ["uhi_tavg", "uhi_tmin"]:
        ss = d[d.income == inc].dropna(subset=[y, "lp"]); m = cl(ss, f"{y} ~ lp"); print(f"  {inc:13s} {y:8s} {ci(m.params['lp'], m.bse['lp'])} n={len(ss)}")

# ---- panel objects ----
exec(open(SP + "/building_volume.py").read().split('print("\\n== A.')[0])
def fe2(dd, rhs, y="uhi_obs"):
    dd = dd.dropna(subset=[y] + rhs.split(" + ")).copy()
    m = smf.ols(f"{y} ~ {rhs} + C(CityID) + C(year)", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd.CityID})
    return " ; ".join(ci(m.params[q], m.bse[q]) + f" p={m.pvalues[q]:.3f}" for q in rhs.split(" + ")) + f" n={dd.CityID.nunique()}"
H = M[M.year >= 2000]
print("== Mundlak within/between by element, 2000-2020 day/night panel")
D = DN.merge(P[["CityID", "year", "ln_popdensity"]].drop_duplicates(["CityID", "year"]), on=["CityID", "year"]); D = D[D.year >= 2000]
print("  columns:", [c for c in D.columns][:20])
def mund(dd, key, y):
    s = dd.dropna(subset=[y, key]).drop_duplicates(["CityID", "year"]).copy()
    for v in [y, key]: s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby("CityID")["t_" + key].transform("mean"); s["b"], s["w"] = bar, s["t_" + key] - bar
    m = smf.ols(f"t_{y} ~ w + b", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.CityID}); return f"within {ci(m.params['w'], m.bse['w'])} | between {ci(m.params['b'], m.bse['b'])} n={s.CityID.nunique()}"
for y in ["uhi_night", "uhi_day", "uhi_mean"]: print(f"  {y:10s} {mund(D, 'ln_popdensity', y)}")
print("== moving-boundary comparison, Supplementary Table S5")
x = pd.read_csv(EXTRA + "ghs_wup_mtuc_r2025a_uc_stats_subset.csv"); x = x[x.Year.isin([2000, 2005, 2010, 2015, 2020])].rename(columns={"Year": "year"}); x["lineage"] = x.ID_MTUC.str.split("_").str[0].astype(int)
cw = pd.concat([pd.read_csv(f).assign(year=int(os.path.basename(f)[-8:-4])) for f in sorted(glob.glob(EXTRA + "mtuc_overlap_crosswalk_*.csv"))])
cw = cw.dropna(subset=["share"]).sort_values("ov_km2", ascending=False).drop_duplicates(["CityID", "year"])
pin = cw[cw.year == 2020][["CityID", "ID_UC_G0"]].rename(columns={"ID_UC_G0": "lineage"}).sort_values("CityID").drop_duplicates("lineage")
mtu = pin.merge(x[["lineage", "year", "AREA_km2", "POP"]], on="lineage"); mtu = mtu[(mtu.AREA_km2 > 0) & (mtu.POP > 0)]
mtu["m_pop"] = np.log(mtu.POP); mtu["m_dens"] = np.log(mtu.POP / mtu.AREA_km2); mtu["m_area"] = np.log(mtu.AREA_km2)
J = H.merge(mtu[["CityID", "year", "m_pop", "m_dens", "m_area"]], on=["CityID", "year"]); print("  joined", J.CityID.nunique(), "cities", len(J), "city-epochs")
print("  frozen on joined  ", fe2(J, "ln_popdensity")); print("  moving population ", fe2(J, "m_pop")); print("  moving density    ", fe2(J, "m_dens"))
print("  density + pop     ", fe2(J, "m_dens + m_pop")); print("  density + area    ", fe2(J, "m_dens + m_area"))
print("== seasonal night level by region, 2000-2020 city means")
sp = pd.read_csv(EXTRA + "seasonal_uhi_panel.csv")
sp["continent"] = sp.continent.fillna(sp.CityID.map(cont))
for ser in ["jja_tmin", "djf_tmin"]:
    cm = sp[(sp.series == ser) & (sp.year >= 2000)].groupby(["CityID", "continent"]).uhi.mean().reset_index()
    for reg in ["North America", "Europe"]: print(f"  {ser} {reg:14s} {mci(cm[cm.continent == reg].uhi)}")
