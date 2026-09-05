"""Table 2 of the main text, the seasonal rows of Supplementary Table S2 and the S1.7 level means, from the seasonal city-epoch panels.
Reads seasonal_uhi_panel.csv (2000-2020) and seasonal_uhi_panel_1975.csv (1975-2020) from UHI_EXTRA_INPUTS."""
import os, sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from supplement_paths import COMP, EXTRA
G = pd.read_csv(COMP + "city_groupings.csv").dropna(subset=["CityID"]).drop_duplicates("CityID"); G["CityID"] = G.CityID.astype(int)
cont = G.set_index("CityID")["continent"].to_dict(); kop = G.set_index("CityID")["koppen_main_group"].to_dict()
def fe(d, y="uhi"):
    d = d.dropna(subset=[y, "ln_popdensity"]).drop_duplicates(["CityID", "year"]).copy()
    if d.CityID.nunique() < 30: return "n/a"
    from linearmodels.panel import PanelOLS
    dd = d.set_index(["CityID", "year"]); k = "ln_popdensity"
    m = PanelOLS(dd[y], dd[[k]], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    return f"{m.params[k]:+.2f} ({m.params[k]-1.96*m.std_errors[k]:+.2f} to {m.params[k]+1.96*m.std_errors[k]:+.2f}) *{d.CityID.nunique()}*"
def load(f):
    sp = pd.read_csv(EXTRA + f); sp["continent"] = sp.continent.fillna(sp.CityID.map(cont)); sp["kop"] = sp.CityID.map(kop); return sp
REG = [("pooled", lambda s: s), ("North America", lambda s: s[s.continent == "North America"]), ("Europe", lambda s: s[s.continent == "Europe"]), ("outside North America", lambda s: s[s.continent.notna() & (s.continent != "North America")])]
for lab, f in [("1975-2020 (Table 2)", "seasonal_uhi_panel_1975.csv"), ("2000-2020 (Supplementary Table S2 rows)", "seasonal_uhi_panel.csv")]:
    sp = load(f); print(f"== {lab}")
    for ser in ["jja_tmin", "djf_tmin", "jja_tavg", "djf_tavg", "jja_tmax", "djf_tmax"]:
        s = sp[sp.series == ser]; print(f"  {ser:9s} | " + " | ".join(fe(g(s)) for _, g in REG))
    # summer minus winter, night
    w = sp[sp.series.isin(["jja_tmin", "djf_tmin"])].pivot_table(index=["CityID", "year"], columns="series", values=["uhi", "ln_popdensity"]).dropna()
    w = pd.DataFrame({"uhi": w["uhi"]["jja_tmin"] - w["uhi"]["djf_tmin"], "ln_popdensity": w["ln_popdensity"]["jja_tmin"]}).reset_index(); w["continent"] = w.CityID.map(cont)
    print("  night, summer minus winter | " + " | ".join(fe(g(w)) for _, g in REG[:3]))
    na = sp[sp.continent == "North America"]
    for ser in ["jja_tmin", "djf_tmin"]:
        print(f"  NA by zone {ser}: " + " / ".join(fe(na[(na.series == ser) & (na.kop == z)]) for z in ["B", "C", "D"]))
    if "2000" in lab:
        print("  city-mean night level, summer / winter (mean, 95% CI):")
        for ser in ["jja_tmin", "djf_tmin"]:
            cm = sp[sp.series == ser].groupby(["CityID", "continent"]).uhi.mean().reset_index()
            for reg in ["North America", "Europe"]:
                x = cm[cm.continent == reg].uhi; se = x.std() / np.sqrt(len(x)); print(f"    {ser} {reg:14s} {x.mean():+.2f} ({x.mean()-1.96*se:+.2f} to {x.mean()+1.96*se:+.2f}) n={len(x)}")
