"""Supplementary Table S2 (the regional gap under each alternative design) and the S1.4 room-to-grow
split, in one script. Prints coefficient, p and n for every row so the table can be checked line by line.

Inputs (set the paths below or the environment variables):
  UHI_AIR_DATA / UHI_AIR_COMPANION   the two Zenodo records (within_city_panel via the code deposit,
                                     city_uhi_epoch_panel_daynight, city_groupings, city_predictors_panel)
  UHI_EXTRA_INPUTS                           folder with gee_built_volume_extraction.csv,
                                     gee_nightlights_extraction_cleaned.csv,
                                     GHS_WUP_MTUC_MT_GLOBE_R2025A_v1_1.xlsx, mtuc_overlap_crosswalk_*.csv,
                                     seasonal_uhi_panel.csv, seasonal_uhi_panel_1975.csv
"""
import os, glob, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, SP)
from supplement_paths import AIR, COMP, EXTRA
exec(open(SP + "/building_volume.py").read().split('print("\\n== A.')[0])   # P, DN, G, V, cont, kop, M, H, D
ctry = G.set_index("CityID")["country"].to_dict(); kf = G.set_index("CityID")["koppen"].to_dict()
M["country"] = M.CityID.map(ctry); M["kopf"] = M.CityID.map(kf); M["kopm"] = M.CityID.map(kop)
H = M[(M.year >= 2000) & M.ln_vol.notna()].copy() if os.environ.get("UHI_VOLUME_SAMPLE", "1") == "1" else M[M.year >= 2000].copy()
def r(m, k): return f"{m.params[k]:+.3f} (p {m.pvalues[k]:.3f}; 95% CI {m.params[k]-1.96*m.bse[k]:+.2f} to {m.params[k]+1.96*m.bse[k]:+.2f})"
def fe2(d, rhs, grp=None, y="uhi_obs"):
    cols = rhs.split(" + "); d = d.dropna(subset=[y] + cols + ([grp] if grp else [])).drop_duplicates(["CityID", "year"]).copy()
    if grp: d["gy"] = d[grp].astype(str) + "_" + d.year.astype(str); f = f"{y} ~ {rhs} + C(CityID) + C(gy)"
    else: f = f"{y} ~ {rhs} + C(CityID) + C(year)"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.CityID})
    return " ; ".join(f"{k} {r(m, k)}" for k in cols) + f"  n={d.CityID.nunique()}"
NA, EU = H[H.continent == "North America"], H[H.continent == "Europe"]
print("== city and year effects, city and year effects"); print("  NA", fe2(NA, "ln_popdensity")); print("  EU", fe2(EU, "ln_popdensity"))
print("== country x year / climate-zone x year effects")
for lab, s in [("NA", NA), ("EU", EU)]:
    print(f"  {lab} country x year ", fe2(s, "ln_popdensity", "country")); print(f"  {lab} Köppen x year  ", fe2(s, "ln_popdensity", "kopf"))
for lab, s in [("NA", M[M.continent == "North America"]), ("EU", M[M.continent == "Europe"])]:
    print(f"  {lab} Köppen x year, 1975-2020", fe2(s, "ln_popdensity", "kopf"))
print("  USA only, Köppen x year", fe2(H[H.country == "USA"], "ln_popdensity", "kopf"))
print("== urban-station distance bins")
cs = pd.read_csv(AIR + "city_station_match_broad.csv").drop_duplicates("city_id")
cid = G.dropna(subset=["city_id"]).drop_duplicates("city_id").set_index("city_id")["CityID"].to_dict()
cs["CityID"] = cs.city_id.map(cid); dist = cs.dropna(subset=["CityID"]).set_index("CityID")["urban_km"]; H["dist"] = H.CityID.map(dist)
for lab, s in [("NA", H[H.continent == "North America"]), ("EU", H[H.continent == "Europe"])]:
    print(f"  {lab} <=5 km ", fe2(s[s.dist <= 5], "ln_popdensity")); print(f"  {lab} 5-15 km", fe2(s[(s.dist > 5) & (s.dist <= 15)], "ln_popdensity"))
print("== population on the moving MTUC boundary")
x = pd.read_csv(EXTRA + "ghs_wup_mtuc_r2025a_uc_stats_subset.csv"); x5 = x[x.Year.isin([2000, 2005, 2010, 2015, 2020])].rename(columns={"Year": "year"}).copy(); x5["lineage"] = x5.ID_MTUC.str.split("_").str[0].astype(int)
cw = pd.concat([pd.read_csv(f).assign(year=int(os.path.basename(f)[-8:-4])) for f in sorted(glob.glob(EXTRA + "mtuc_overlap_crosswalk_*.csv"))])
cw = cw.dropna(subset=["share"]).sort_values("ov_km2", ascending=False).drop_duplicates(["CityID", "year"])
pin = cw[cw.year == 2020][["CityID", "ID_UC_G0"]].rename(columns={"ID_UC_G0": "lineage"}).sort_values("CityID").drop_duplicates("lineage")
mt = pin.merge(x5[["lineage", "year", "AREA_km2", "POP"]], on="lineage"); mt = mt[(mt.AREA_km2 > 0) & (mt.POP > 0)]; mt["m_pop"] = np.log(mt.POP)
J = H.merge(mt[["CityID", "year", "m_pop"]], on=["CityID", "year"])
print("  NA", fe2(J[J.continent == "North America"], "m_pop")); print("  EU", fe2(J[J.continent == "Europe"], "m_pop"))
print("== building volume added"); print("  NA", fe2(NA, "ln_popdensity + ln_vol")); print("  EU", fe2(EU, "ln_popdensity + ln_vol"))
print("== Köppen main group only")
for z, lab in [("C", "temperate"), ("D", "continental"), ("B", "arid")]:
    print(f"  NA {lab:11s}", fe2(NA[NA.kopm == z], "ln_popdensity"));
    if z != "B": print(f"  EU {lab:11s}", fe2(EU[EU.kopm == z], "ln_popdensity"))
print("== night-time lights, within-sensor variation (city x sensor effects)")
n = pd.read_csv(EXTRA + "gee_nightlights_extraction_cleaned.csv"); n = n[~n.zone.astype(str).str.startswith("rural")].drop_duplicates(["CityID", "year"]); n["ln_ntl"] = np.log(n.NIGHTLIGHTS_value.clip(lower=0) + 1e-3)
JN = H.merge(n[["CityID", "year", "ln_ntl", "sensor"]], on=["CityID", "year"]); JN["cs"] = JN.CityID.astype(str) + "_" + JN.sensor
def fecs(d, rhs, y="uhi_obs"):
    d = d.dropna(subset=[y] + rhs.split(" + ")).copy(); d = d[d.groupby("cs").year.transform("nunique") >= 2]
    m = smf.ols(f"{y} ~ {rhs} + C(cs) + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.CityID})
    return " ; ".join(f"{k} {r(m, k)}" for k in rhs.split(" + ")) + f"  n={d.CityID.nunique()}"
print("  NA", fecs(JN[JN.continent == "North America"], "ln_popdensity + ln_ntl")); print("  EU", fecs(JN[JN.continent == "Europe"], "ln_popdensity + ln_ntl"))
print("== reference annulus crowding (all MTUC centres 40-150 km) and 40-500 km landscape settlement")
x20 = x[x.Year == 2020]; lat = np.radians(x20.Lat.values.astype(float)); lon = np.radians(x20.Lon.values.astype(float)); pop = x20.POP.fillna(0).values.astype(float)
cp = pd.read_csv(COMP + "city_predictors_panel.csv"); c = cp[cp.year == 2020].drop_duplicates("CityID")[["CityID", "lat", "lon"]].dropna(); c = c[c.CityID.isin(H.CityID)]
def band(la, lo, r0, r1):
    d = 6371 * np.arccos(np.clip(np.sin(la) * np.sin(lat) + np.cos(la) * np.cos(lat) * np.cos(lon - lo), -1, 1)); m = (d >= r0) & (d <= r1)
    return int(m.sum()), float(pop[m].sum()), float(pop[m].sum() / (np.pi * (r1 ** 2 - r0 ** 2)))
rows = []
for q in c.itertuples():
    la, lo = np.radians(q.lat), np.radians(q.lon); n_ring, pop_ring, _ = band(la, lo, 40, 150); _, _, d500 = band(la, lo, 40, 500)
    rows.append((q.CityID, n_ring, pop_ring, d500))
cr = pd.DataFrame(rows, columns=["CityID", "n_ring", "pop_ring", "dens500"]); cr["ln_popring"] = np.log(cr.pop_ring + 1); cr["ln_d500"] = np.log(cr.dens500 + 1e-3)
JA = H.merge(cr, on="CityID"); J1 = JA.drop_duplicates("CityID")
print("  premise:", J1.groupby("continent").agg(n_ring=("n_ring", "median"), ringpop_M=("pop_ring", lambda v: round(v.median() / 1e6, 1)), dens500=("dens500", "median")).round(1).to_dict("index"))
for lab, s in [("NA", JA[JA.continent == "North America"]), ("EU", JA[JA.continent == "Europe"])]:
    qq = s.drop_duplicates("CityID").pop_ring.quantile([1 / 3, 2 / 3]).values
    print(f"  {lab} annulus quiet  ", fe2(s[s.pop_ring <= qq[0]], "ln_popdensity")); print(f"  {lab} annulus middle ", fe2(s[(s.pop_ring > qq[0]) & (s.pop_ring <= qq[1])], "ln_popdensity")); print(f"  {lab} annulus crowded", fe2(s[s.pop_ring > qq[1]], "ln_popdensity"))
    t = s.copy(); t["x"] = t.ln_popdensity * (t.ln_popring - t.ln_popring.mean()); m = smf.ols("uhi_obs ~ ln_popdensity + x + C(CityID) + C(year)", data=t).fit(cov_type="cluster", cov_kwds={"groups": t.CityID}); print(f"  {lab} interaction with ln annulus population", r(m, "x"))
qm = J1.dens500.median(); print(f"  pooled median landscape settlement {qm:.1f}/km2")
for lab, s in [("NA", JA[JA.continent == "North America"]), ("EU", JA[JA.continent == "Europe"])]:
    print(f"  {lab} landscape sparse ", fe2(s[s.dens500 <= qm], "ln_popdensity")); print(f"  {lab} landscape settled", fe2(s[s.dens500 > qm], "ln_popdensity"))
    t = s.copy(); t["x"] = t.ln_popdensity * (t.ln_d500 - t.ln_d500.mean()); m = smf.ols("uhi_obs ~ ln_popdensity + x + C(CityID) + C(year)", data=t).fit(cov_type="cluster", cov_kwds={"groups": t.CityID}); print(f"  {lab} interaction with ln landscape settlement", r(m, "x"))
s = JA[JA.continent == "North America"]; qq = s.drop_duplicates("CityID").dens500.quantile([1 / 3, 2 / 3]).values
print("  NA landscape terciles: sparse", fe2(s[s.dens500 <= qq[0]], "ln_popdensity"), "| settled", fe2(s[s.dens500 > qq[1]], "ln_popdensity"))
print("== seasonal night term, summer / winter")
GH = EXTRA
for lab, f in [("2000-2020", "seasonal_uhi_panel.csv"), ("1975-2020", "seasonal_uhi_panel_1975.csv")]:
    sp = pd.read_csv(GH + f); sp["continent"] = sp.continent.fillna(sp.CityID.map(cont))
    for ser in ["jja_tmin", "djf_tmin"]:
        for reg in ["North America", "Europe"]:
            t = sp[(sp.series == ser) & (sp.continent == reg)]; print(f"  {lab} {ser} {reg:14s}", fe2(t, "ln_popdensity", y="uhi"))
print("== S1.4 room to grow, high-income cities, 1975-2020")
inc = G.set_index("CityID")["income"].to_dict(); M["income"] = M.CityID.map(inc)
g = M.groupby("CityID").agg(built=("frac_built", "mean"), growth=("ln_popdensity", lambda v: v.max() - v.min()), income=("income", "first")); hi = g[g.income == "High"]
qb = hi.built.quantile([1 / 3, 2 / 3]).values; print("  high-income cities:", len(hi))
for lab, ids in [("low built (room)", hi[hi.built <= qb[0]].index), ("middle", hi[(hi.built > qb[0]) & (hi.built <= qb[1])].index), ("high built (built out)", hi[hi.built > qb[1]].index)]:
    print(f"  {lab:24s}", fe2(M[M.CityID.isin(ids)], "ln_popdensity"))
for bl, bids in [("room", hi.built <= hi.built.median()), ("built out", hi.built > hi.built.median())]:
    for gl, gids in [("stalled", hi.growth <= hi.growth.median()), ("grew", hi.growth > hi.growth.median())]:
        print(f"  {bl:9s} x {gl:8s}", fe2(M[M.CityID.isin(hi[bids & gids].index)], "ln_popdensity"))
