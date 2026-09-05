"""Historical Stage-3 on station UHI with clean-rural referencing, two-way FE.
Model A: 1975-2020 (10 epochs) density + built-up.
Model B: 1990-2020 (7 epochs) density + RCS(lnGDPpc) + built-up.
Predictors from local GHSL(POP,BUS)+area+GDP_SUM; DV=clean-rural station UHI (5-yr window at each epoch).
Compare to the 2000-2020-only model to show the payoff of extra within-city urbanization variation.

Rural referencing
-----------------
The reference was previously re-selected at every epoch, from whichever of a city's clean-rural
stations had data in that epoch's five-year window. A within-city change in UHI then mixes a real
change with a change in who the reference is, because a station entering or leaving moves the
median. The drift is negligible over twenty years, where the set barely turns over, and severe
over forty-five, where it does: on 1975/80 -> 2015/20 the two constructions correlate at only
r = 0.54 and carry opposite signs.

Three constructions are available; `anomaly` is the default.

  anomaly   the reference time profile comes from a least-squares fit of T[s,t] = alpha_s + tau_t
            over ALL of the city's clean-rural stations. alpha_s absorbs each station's level
            (elevation, exposure, siting) and tau_t is the common time signal, identified from
            overlaps rather than from whoever happens to be present. Nothing is discarded and the
            profile is composition free. The reference is R(t) = median_s(alpha_s) + tau_t, whose
            constant part is absorbed by city fixed effects in the within estimator. The
            median matches the level convention of the median-of-stations estimator it
            replaces, so levels stay comparable; a mean would inflate them by ~0.5 C.
  fixed     the reference is restricted to stations valid at every epoch the city contributes.
            Composition is constant by construction, at the cost of discarding stations and
            cities, and the panel has to be rebuilt per window.
  varying   the previous construction, retained for comparison.

The two corrections are independent -- one discards stations, the other models them -- and they
agree with each other while both disagree with `varying`, which is what identifies the drift as
real rather than an artifact of either method. `anomaly` keeps every city, uses more references,
and carries the smallest standard errors, so it is the default. Its within component is also
window invariant (r = 0.998 between a profile fitted on ten epochs and one fitted on five), so a
single ten-epoch panel serves every estimation window.

Everything else is unchanged: 12 km clean-rural screen, five-year window with >=3 valid years,
>=3 references, 6.5 C/km on urban-minus-rural-median elevation."""
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from linearmodels.panel import PanelOLS

# Paths resolve relative to this file, or from the environment, so the scripts run from a clone
# without editing. UHI_AIR_DATA points at the unpacked GHCN-Daily station UHI release (the data descriptor's
# Zenodo deposit); UHI_AIR_COMPANION at the companion deposit (predictors and satellite panel).
# Everything this script needs is in those two.
import os as _os
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
RELEASE = str(_P(_os.environ.get("UHI_AIR_DATA", _HERE.parent / "data" / "air_record")))
INPUTS  = str(_HERE.parent / "data" / "inputs")
COMPANION = str(_P(_os.environ.get("UHI_AIR_COMPANION", _HERE.parent / "data" / "companion")))

EP = [1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
GY = [1990,1995,2000,2005,2010,2015,2020]
# ---- predictors ----
# hist_predictors.csv already carries ln_popdensity, frac_built, ln_gdp_c and the g1-g3 spline
# basis, built once from the raw GHS-UCDB GeoPackages (population, built-up area, GDP sum) by the
# companion deposit's own construction. Reading it here rather than rebuilding from the GeoPackages
# keeps this script reproducible from the two public deposits alone, with no GHS files to fetch
# separately.
pred = pd.read_csv(f"{COMPANION}/hist_predictors.csv")
# ---- DV: clean-rural station UHI at each epoch ----
meta=pd.read_csv(f"{RELEASE}/need_broad_meta.csv",dtype={"id":str}); S={r.id:(r.lat,r.lon,r.elev) for r in meta.itertuples()}
adf=pd.read_csv(f"{RELEASE}/annual_by_elem.csv",dtype={"id":str}); TA={}
for r in adf.itertuples():
    if pd.notna(r.tavg): TA.setdefault(r.id,{})[r.year]=r.tavg
match=pd.read_csv(f"{RELEASE}/city_station_match_broad.csv",dtype={"urban":str,"rural":str})
# The "is this rural station near ANY city" screen used to check distance against a legacy
# ring-panel file; city_centroids.csv, now deposited with the release, does the same job and is
# the list the release's own 12 km contamination screen is built against (Sect. 2.4).
allc=pd.read_csv(f"{RELEASE}/city_centroids.csv")[["lon","lat"]].drop_duplicates()
def proj(la,lo): return np.c_[la*111.0,lo*111.0*np.cos(np.radians(la))]
ct=cKDTree(proj(allc.lat.values,allc.lon.values)); cc={}
def clean(s):
    if s in cc: return cc[s]
    la,lo,_=S[s]; d,_=ct.query(proj(np.array([la]),np.array([lo]))); cc[s]=d[0]>=12.0; return cc[s]
L=6.5/1000
def wm(sid,a,b,mn=3):
    d=TA.get(sid,{}); x=[d[y] for y in range(a,b+1) if y in d]; return np.mean(x) if len(x)>=mn else np.nan
def tau_profile(stations, yrs):
    """Least squares alpha_s + tau_t on the rural station-by-epoch matrix; returns R(t)."""
    obs=[(si,ti,wm(st,y-2,y+2)) for si,st in enumerate(stations) for ti,y in enumerate(yrs)
         if not np.isnan(wm(st,y-2,y+2))]
    if not obs: return None
    ns,nt=len(stations),len(yrs)
    X=np.zeros((len(obs),ns+nt-1)); v=np.zeros(len(obs))
    for i,(si,ti,val) in enumerate(obs):
        X[i,si]=1.0
        if ti>0: X[i,ns+ti-1]=1.0
        v[i]=val
    beta,*_=np.linalg.lstsq(X,v,rcond=None)
    alpha=beta[:ns]; tau=np.r_[0.0,beta[ns:]]
    cnt={}
    for _,ti,_ in obs: cnt[ti]=cnt.get(ti,0)+1
    base=float(np.median(alpha))
    return {yrs[ti]: base+tau[ti] for ti in range(nt) if cnt.get(ti,0)>=3}

def build_dv(epochs, mode="anomaly"):
    """UHI at each epoch. mode in {"anomaly","fixed","varying"}; see module docstring."""
    rows=[]
    for r in match.itertuples():
        u=r.urban
        if u not in TA: continue
        cl=[s for s in (r.rural.split(";") if isinstance(r.rural,str) else []) if s in TA and clean(s)]
        if len(cl)<3: continue
        ue=S[u][2]
        yrs=[y for y in epochs if not np.isnan(wm(u,y-2,y+2))]
        if not yrs: continue
        if mode=="anomaly":
            R=tau_profile(cl,yrs)
            if not R: continue
            re0=np.nanmedian([S[s][2] for s in cl])
            ec=L*(ue-re0) if np.isfinite(ue) and np.isfinite(re0) else 0.0
            for y in yrs:
                if y in R:
                    rows.append({"CityID":r.city_id,"year":y,"n_ref":len(cl),
                                 "uhi_obs":wm(u,y-2,y+2)-R[y]+ec})
            continue
        if mode=="fixed":
            rv0=[s for s in cl if all(not np.isnan(wm(s,y-2,y+2)) for y in yrs)]
            if len(rv0)<3: continue
            re0=np.nanmedian([S[s][2] for s in rv0])
            ec0=L*(ue-re0) if np.isfinite(ue) and np.isfinite(re0) else 0.0
        for y in yrs:
            a,b=y-2,y+2; uw=wm(u,a,b)
            if mode=="fixed": rv,ec=rv0,ec0
            else:
                rv=[s for s in cl if not np.isnan(wm(s,a,b))]
                if len(rv)<3: continue
                re=np.nanmedian([S[s][2] for s in rv])
                ec=L*(ue-re) if np.isfinite(ue) and np.isfinite(re) else 0.0
            rows.append({"CityID":r.city_id,"year":y,"n_ref":len(rv),
                         "uhi_obs":uw-np.median([wm(s,a,b) for s in rv])+ec})
    return pd.DataFrame(rows).merge(pred,on=["CityID","year"],how="inner")

d=build_dv(EP,"anomaly")   # the default construction
def fit(dd,cols,tag):
    dd=dd.dropna(subset=["uhi_obs"]+cols).drop_duplicates(["CityID","year"]).set_index(["CityID","year"])
    nc=dd.index.get_level_values(0).nunique()
    m=PanelOLS(dd["uhi_obs"],dd[cols],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type="clustered",cluster_entity=True)
    print(f"\n--- {tag}: n={int(m.nobs):,} city-epochs, {nc} cities, epochs {sorted(dd.index.get_level_values(1).unique())} ---")
    for t in cols:
        if t in ("g1","g2","g3"): continue
        print(f"    {t:16} {m.params[t]:+.3f} (se {m.std_errors[t]:.3f}, p {m.pvalues[t]:.1e})")
print(f"clean-rural DV panel: {len(d):,} city-epochs, {d.CityID.nunique()} cities")
print("satellite targets: ln_popdensity +0.154, frac_urban_built +1.127")

MODELS=[("Baseline 2000-2020 (density+built)", [y for y in EP if y>=2000], ["ln_popdensity","frac_built"]),
        ("Model A: 1975-2020 (density+built, 10 epochs)", EP, ["ln_popdensity","frac_built"]),
        ("Model B: 1990-2020 FULL (density+RCS GDP+built)", [y for y in EP if y>=1990],
         ["ln_popdensity","g1","g2","g3","frac_built"])]
for tag,epochs,cols in MODELS:
    for mode in ("anomaly","fixed","varying"):
        fit(build_dv(epochs,mode),cols,f"{tag}  [{mode} reference]")

OUT=f"{INPUTS}/"
for mode,name in (("anomaly","hist_stage3_panel.csv"),
                  ("fixed","hist_stage3_panel_fixed.csv"),
                  ("varying","hist_stage3_panel_varying.csv")):
    dd=build_dv(EP,mode)
    dd.to_csv(OUT+name,index=False)
    print(f"saved {name}: {len(dd):,} city-epochs, {dd.CityID.nunique():,} cities -> {OUT}")
print("\nhist_stage3_panel.csv is now the anomaly-referenced panel; the previous construction is")
print("kept alongside it as hist_stage3_panel_varying.csv.")
