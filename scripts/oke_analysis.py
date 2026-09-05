"""
Oke population-size law + level-vs-trend density structure — full reproducible analysis.
Regenerates every result CSV in ../data/results/ from ../data/inputs/.
Self-locating paths (run from anywhere).  Papers 1 & 2 (air-temperature UHI).

Sections:
  A  master city table (+ daytime uhi_max = 2*tavg - tmin, exact under TAVG=(TMAX+TMIN)/2)
  B  Oke size-law univariate fits, day/mean/night           -> oke_size_law_fits.csv
  C  size-law by climate zone (night)                        -> oke_size_law_by_climate_zone.csv
  D  incremental covariate ladder (night/mean/day)           -> oke_incremental_covariates{,_MEAN,_MAX}.csv
  E  station-distance sensitivity (night/mean/day)           -> oke_station_distance_sensitivity{,_MEAN,_MAX}.csv
  F  development phase (size + temporal)                     -> oke_size_law_by_development_phase.csv
  G  literature recalibration (Oke1973/Karl1988) to our form -> oke_literature_{recalibration,matched_coverage}.csv
  H  coverage-bias partial correction (capped-60%)           -> coverage_corrected_uhi_premium.csv
  I  between-city density checks                            -> density_{between_element,incremental_covariates,by_climate_zone}.csv
  J  within-city density (panel, 2000-2020 observation era)  -> density_within_city_sensitivity.csv
  K  window sweeps + coverage-by-year                       (printed)
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm, statsmodels.formula.api as smf
try:
    from linearmodels import PanelOLS
    HAVE_PANEL = True
except Exception:
    HAVE_PANEL = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IN   = os.path.join(ROOT, "data", "inputs") + os.sep
OUT  = os.path.join(ROOT, "data", "results") + os.sep
os.makedirs(OUT, exist_ok=True)

# ---------- A. master city table ----------
pr   = pd.read_csv(IN+"city_uhi_predictors.csv")
pr["uhi_max"] = 2*pr.uhi_tavg - pr.uhi_tmin                       # daytime UHI (exact identity)
rep  = pd.read_csv(IN+"representativeness.csv")[["CityID","pop","abslat","income","latband","kop","sampled"]]
mtch = pd.read_csv(IN+"city_station_match.csv")[["city_id","urban_km","n_rural"]]
lst  = pd.read_csv(IN+"uhi_panel_koppen_final_reconstructed.csv")
lstg = lst.groupby("CityID").agg(u_ELEV=("u_ELEV","first"), coast=("coastal_dist_km","first")).reset_index()
d = pr.merge(rep, on="CityID", how="left").merge(mtch, on="city_id", how="left").merge(lstg, on="CityID", how="left")
# the level sample is the cities carrying analysis covariates; without this restriction the
# fits run on ~37 extra cities and none of the three element slopes matches the reported table
d = d.dropna(subset=["ln_popdensity"])
d = d[d["pop"]>0].copy()
d["lp"] = np.log10(d["pop"]); d["log10_pop"] = d["lp"]; d["kg"] = d.koppen.astype(str).str[0]
grp = lambda df: pd.factorize(df.country)[0]
def cl(df, formula):     # OLS with country-clustered SE (or HC1 if single cluster)
    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups":grp(df)}) if df.country.nunique()>1 \
           else smf.ols(formula, data=df).fit(cov_type="HC1")

# ---------- B. Oke size-law univariate fits ----------
rows=[]
for y,lab in [("uhi_max","daytime (TMAX)"),("uhi_tavg","mean (TAVG)"),("uhi_tmin","nighttime (TMIN)")]:
    s=d.dropna(subset=[y,"lp"]); m=cl(s,f"{y} ~ lp")
    rows.append([lab,round(m.params["Intercept"],3),round(m.params["lp"],3),round(m.bse["lp"],3),
                 round(m.pvalues["lp"],4),round(m.rsquared,3),round(s[y].mean(),3),int(m.nobs)])
# The three elements are fitted on whichever cities resolve each, so their slopes satisfy
# day = 2*mean - night only on a common sample. Table 1 reports the mean law on both, and this
# row is the one that closes the identity; see Section 4.4.
common=d.dropna(subset=["uhi_tavg","uhi_tmin","lp"])
mc=cl(common,"uhi_tavg ~ lp")
rows.append(["mean (TAVG), common sample",round(mc.params["Intercept"],3),round(mc.params["lp"],4),
             round(mc.bse["lp"],3),round(mc.pvalues["lp"],4),round(mc.rsquared,3),
             round(common.uhi_tavg.mean(),3),int(mc.nobs)])
_n=cl(common,"uhi_tmin ~ lp").params["lp"]; _dd=cl(common,"uhi_max ~ lp").params["lp"]
print(f"  identity on the common sample: 2*({mc.params['lp']:+.4f}) - ({_n:+.4f}) = "
      f"{2*mc.params['lp']-_n:+.4f}  against the fitted day {_dd:+.4f}")
pd.DataFrame(rows,columns=["uhi_measure","intercept","slope_per_log10pop","slope_SE","slope_p","R2","mean_uhi","n"]).to_csv(OUT+"oke_size_law_fits.csv",index=False)

# ---------- C. size-law by climate zone (night) ----------
zr=[]
for k,lab in [("A","tropical"),("B","arid"),("C","temperate"),("D","continental")]:
    s=d[d.kg==k].dropna(subset=["uhi_tmin","lp"]); m=cl(s,"uhi_tmin ~ lp")
    zr.append([k,lab,round(m.params["lp"],3),round(m.bse["lp"],3),round(m.pvalues["lp"],3),round(s.uhi_tmin.mean(),2),len(s)])
pd.DataFrame(zr,columns=["koppen_group","climate","night_size_slope","SE","p","mean_night_uhi","n"]).to_csv(OUT+"oke_size_law_by_climate_zone.csv",index=False)

# ---------- D & E. incremental ladder + station distance, per element ----------
ALLV=["lp","koppen","abslat","income","ln_gdp_c","ln_popdensity","frac_urban_built","u_ELEV","coast","urban_km","n_rural","country"]
LADDER=[("M0 size only",[]),("M1 +climate zone",["koppen"]),("M2 +abs latitude",["koppen","abslat"]),
 ("M3 +income",["koppen","abslat","income"]),("M4 +GDP/capita",["koppen","abslat","income","ln_gdp_c"]),
 ("M5 +pop density",["koppen","abslat","income","ln_gdp_c","ln_popdensity"]),
 ("M6 +built fraction",["koppen","abslat","income","ln_gdp_c","ln_popdensity","frac_urban_built"]),
 ("M7 +elevation",["koppen","abslat","income","ln_gdp_c","ln_popdensity","frac_urban_built","u_ELEV"]),
 ("M8 +coastal distance",["koppen","abslat","income","ln_gdp_c","ln_popdensity","frac_urban_built","u_ELEV","coast"]),
 ("M9 +station siting",["koppen","abslat","income","ln_gdp_c","ln_popdensity","frac_urban_built","u_ELEV","coast","urban_km","n_rural"])]
def element_checks(Y, suffix):
    S=d.dropna(subset=[Y]+ALLV).reset_index(drop=True); g=pd.factorize(S.country)[0]; y=S[Y].values
    kd=pd.get_dummies(S.koppen.astype(str),prefix="k",drop_first=True).astype(float)
    idd=pd.get_dummies(S.income.astype(str),prefix="inc",drop_first=True).astype(float)
    def cols(names):
        parts=[S[["lp"]].astype(float)]
        for nm in names: parts.append(kd if nm=="koppen" else idd if nm=="income" else S[[nm]].astype(float))
        return sm.add_constant(pd.concat(parts,axis=1))
    ir=[]
    for lbl,nm in LADDER:
        X=cols(nm); m=sm.OLS(y,X.values).fit(cov_type="cluster",cov_kwds={"groups":g}); i=list(X.columns).index("lp")
        ir.append([lbl,round(m.params[i],3),round(m.bse[i],3),round(m.pvalues[i],4),round(m.rsquared,3),len(S)])
    pd.DataFrame(ir,columns=["model","size_slope","SE","p","R2","n"]).to_csv(OUT+f"oke_incremental_covariates{suffix}.csv",index=False)
    dr=[]; X=sm.add_constant(S[["lp","urban_km"]].astype(float)); m=sm.OLS(y,X.values).fit(cov_type="cluster",cov_kwds={"groups":g})
    dr.append(["urban_km coef (per km)",round(m.params[2],4),round(m.pvalues[2],3),"",len(S)])
    for lo,hi,lab in [(0,5,"size-slope, station <=5km"),(5,15,"size-slope, 5-15km"),(15,26,"size-slope, >15km")]:
        s=S[(S.urban_km>=lo)&(S.urban_km<hi)]; gg=pd.factorize(s.country)[0]
        mm=sm.OLS(s[Y].values,sm.add_constant(s[["lp"]].astype(float)).values).fit(cov_type="cluster",cov_kwds={"groups":gg})
        dr.append([lab,round(mm.params[1],3),round(mm.pvalues[1],3),round(s[Y].mean(),2),len(s)])
    Xi=S[["lp","urban_km"]].astype(float).copy(); Xi["ix"]=Xi.lp*Xi.urban_km
    mi=sm.OLS(y,sm.add_constant(Xi).values).fit(cov_type="cluster",cov_kwds={"groups":g})
    dr.append(["size x distance interaction",round(mi.params[3],4),round(mi.pvalues[3],3),"",len(S)])
    for lo in [1,3,5]:
        s=S[S.n_rural>=lo]; gg=pd.factorize(s.country)[0]
        mm=sm.OLS(s[Y].values,sm.add_constant(s[["lp"]].astype(float)).values).fit(cov_type="cluster",cov_kwds={"groups":gg})
        dr.append([f"size-slope, rural-ref >={lo}",round(mm.params[1],3),round(mm.pvalues[1],3),"",len(s)])
    pd.DataFrame(dr,columns=["test","estimate","p","mean_uhi","n"]).to_csv(OUT+f"oke_station_distance_sensitivity{suffix}.csv",index=False)
for Y,suf in [("uhi_tmin",""),("uhi_tavg","_MEAN"),("uhi_max","_MAX")]:
    element_checks(Y,suf)

# ---------- F. development phase (size + temporal) ----------
def sl(sub,y):
    s=sub.dropna(subset=[y,"lp"]);  return (np.nan,np.nan,len(s)) if len(s)<25 else \
        (lambda m:(round(m.params["lp"],3),round(m.pvalues["lp"],4),len(s)))(cl(s,f"{y} ~ lp"))
rows=[]
for g_,phase in [("Upper-middle","actively urbanizing"),("High","mature")]:
    sub=d[d.income==g_]
    for y,t in [("uhi_tmin","night"),("uhi_tavg","mean")]:
        b,p,n=sl(sub,y); rows.append(["income-stratified",g_,phase,t,b,p,n,round(sub.uhi_tavg.mean(),2)])
pn=pd.read_csv(IN+"hist_stage3_panel.csv").dropna(subset=["uhi_obs"]).merge(rep[["CityID","pop","income"]],on="CityID",how="left")
pn=pn.merge(pr[["CityID","country"]],on="CityID",how="left"); pn=pn[pn["pop"]>0].copy(); pn["lp"]=np.log10(pn["pop"])
for lab,sub in [("early 1975-1997",pn[pn.year<=1997]),("recent 2006-2020",pn[pn.year>=2006])]:
    cs=sub.groupby("CityID").agg(u=("uhi_obs","mean"),lp=("lp","first"),country=("country","first")).reset_index().dropna()
    m=smf.ols("u ~ lp",data=cs).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(cs.country)[0]})
    rows.append(["temporal",lab,"maturation over time","mean",round(m.params["lp"],3),round(m.pvalues["lp"],4),len(cs),np.nan])
rows.append(["literature ref","Karl 1988 (US mid-century)","actively urbanizing","annual-avg",0.586,np.nan,"US",np.nan])
pd.DataFrame(rows,columns=["stratification","group","phase","uhi_time","size_slope_per_tenfold","p","n","mean_uhi_level"]).to_csv(OUT+"oke_size_law_by_development_phase.csv",index=False)

# ---------- G. literature recalibration (Oke 1973 & Karl 1988/Estrada) to our log-linear form ----------
mk=np.polyfit(d.lp.values, 0.00174*d["pop"].values**0.45, 1)[0]     # Karl power law -> local log-linear slope on our pop dist
rec=[["Karl et al. 1988 (Estrada src)","a*P^0.45 (power)","annual AVG","air","United States",round(mk,3),round(np.polyfit(d.lp.values,0.00174*d["pop"].values**0.45,1)[1]+mk*6,2)],
     ["Oke 1973 N.America","2.96*log10P-6.41","MAX (calm-clear night)","canopy air","Canada/N.Am",2.96,round(2.96*6-6.41,2)],
     ["Oke 1973 Europe","2.01*log10P-4.06","MAX (calm-clear night)","canopy air","Europe",2.01,round(2.01*6-4.06,2)]]
for y,lab in [("uhi_tmin","OURS night"),("uhi_tavg","OURS mean"),("uhi_max","OURS day")]:
    s=d.dropna(subset=[y,"lp"]); mm=smf.ols(f"{y} ~ lp",data=s).fit()
    rec.append([lab,"c+m*log10P (fit)","annual "+lab.split()[-1],"air","global (dev-world)",round(mm.params["lp"],3),round(mm.params["Intercept"]+mm.params["lp"]*6,2)])
pd.DataFrame(rec,columns=["source","native_form","uhi_time_type","temp","coverage","slope_per_tenfold_ourform","pred_at_1M"]).to_csv(OUT+"oke_literature_recalibration.csv",index=False)
# matched-coverage (region-restricted our slopes vs matched literature)
US=d[d.country.isin(["US","USA","United States"])]; NA=d[d.continent=="North America"]; EU=d[d.continent=="Europe"]
mc=[]
for reg,sub in [("United States",US),("North America",NA),("Europe",EU),("Global (all)",d)]:
    for y,t in [("uhi_tmin","night"),("uhi_tavg","mean"),("uhi_max","day")]:
        s=sub.dropna(subset=[y,"lp"]); m=cl(s,f"{y} ~ lp") if len(s)>=20 else None
        mc.append(["OURS (GHCN-Daily)",reg,t,"annual, all-weather","station air",round(m.params["lp"],3) if m is not None else np.nan,len(s)])
mc+=[["Karl et al. 1988 (=Estrada src)","United States","annual avg","annual, all-weather","station air",round(mk,3),"US"],
     ["Oke 1973","North America","MAX (nocturnal)","calm-clear","canopy air",2.96,"-"],
     ["Oke 1973","Europe","MAX (nocturnal)","calm-clear","canopy air",2.01,"-"]]
pd.DataFrame(mc,columns=["source","region","uhi_time","weather","temp_type","slope_per_tenfold_ourform","n"]).to_csv(OUT+"oke_literature_matched_coverage.csv",index=False)

# ---------- H. coverage-bias partial correction (capped-60%) ----------
cc=d.copy(); samp=cc[cc.sampled==1].copy(); raw=samp.uhi_tavg.mean()
def capped60(stratcols,cap=95,frac=0.6):
    cc["st"]=cc[stratcols].astype(str).agg("|".join,axis=1); samp["st"]=samp[stratcols].astype(str).agg("|".join,axis=1)
    g=cc.groupby("st")["pop"].sum(); g=g/g.sum(); ss=samp.groupby("st")["pop"].sum(); ss=ss/ss.sum()
    w=samp["st"].map(g/ss).replace([np.inf,-np.inf],np.nan); sv=samp.assign(w=w).dropna(subset=["uhi_tavg","w"])
    wc=np.minimum(sv["w"].values,np.percentile(sv["w"].values,cap)); full=np.average(sv["uhi_tavg"],weights=wc)
    ess=(wc.sum()**2)/(wc**2).sum(); return round(full,3), round(raw+frac*(full-raw),3), round(ess,0)
hr=[]
for cols,name in [(["latband","kop"],"latband x koppen"),(["income","continent"],"income x continent"),(["latband","kop","income"],"latband x koppen x income")]:
    full,c60,ess=capped60(cols); hr.append([name,full,c60,ess])
hr.append(["RAW_SAMPLE_MEAN",round(raw,3),round(raw,3),len(samp)])
hr.append(["CANONICAL_capped60_mean","",round(np.mean([r[2] for r in hr[:3]]),3),""])
pd.DataFrame(hr,columns=["raking_scheme","capped_full_uhi_C","capped60_uhi_C","effective_sample_size"]).to_csv(OUT+"coverage_corrected_uhi_premium.csv",index=False)

# ---------- I. between-city density checks ----------
D="ln_popdensity"; er=[]
for y,t in [("uhi_tmin","night"),("uhi_tavg","mean"),("uhi_max","day")]:
    s=d.dropna(subset=[y,D,"lp"]); m1=cl(s,f"{y} ~ {D}"); m2=cl(s,f"{y} ~ {D} + lp")
    er.append([t,round(m1.params[D],3),round(m1.pvalues[D],3),round(m2.params[D],3),round(m2.pvalues[D],3),len(s)])
pd.DataFrame(er,columns=["uhi_time","density_only","p","density_plus_size","p_ctrl","n"]).to_csv(OUT+"density_between_element.csv",index=False)
S=d.dropna(subset=["uhi_tmin",D]+ALLV).reset_index(drop=True); g=pd.factorize(S.country)[0]; y=S.uhi_tmin.values
kd=pd.get_dummies(S.koppen.astype(str),prefix="k",drop_first=True).astype(float); idd=pd.get_dummies(S.income.astype(str),prefix="i",drop_first=True).astype(float)
def dcols(names):
    parts=[S[[D]].astype(float)]
    for nm in names: parts.append(kd if nm=="koppen" else idd if nm=="income" else S[[nm]].astype(float))
    return sm.add_constant(pd.concat(parts,axis=1))
DL=[("M0 density only",[]),("M1 +climate",["koppen"]),("M2 +latitude",["koppen","abslat"]),("M3 +income",["koppen","abslat","income"]),
 ("M4 +GDP",["koppen","abslat","income","ln_gdp_c"]),("M5 +POPULATION",["koppen","abslat","income","ln_gdp_c","lp"]),
 ("M6 +built",["koppen","abslat","income","ln_gdp_c","lp","frac_urban_built"]),("M7 +elevation",["koppen","abslat","income","ln_gdp_c","lp","frac_urban_built","u_ELEV"]),
 ("M8 +coastal",["koppen","abslat","income","ln_gdp_c","lp","frac_urban_built","u_ELEV","coast"]),
 ("M9 +siting",["koppen","abslat","income","ln_gdp_c","lp","frac_urban_built","u_ELEV","coast","urban_km","n_rural"])]
ir=[]
for lbl,nm in DL:
    X=dcols(nm); m=sm.OLS(y,X.values).fit(cov_type="cluster",cov_kwds={"groups":g}); i=list(X.columns).index(D)
    ir.append([lbl,round(m.params[i],3),round(m.bse[i],3),round(m.pvalues[i],4),round(m.rsquared,3)])
pd.DataFrame(ir,columns=["model","density_slope","SE","p","R2"]).to_csv(OUT+"density_incremental_covariates.csv",index=False)
zr=[]
for k,lab in [("A","tropical"),("B","arid"),("C","temperate"),("D","continental")]:
    s=d[d.kg==k].dropna(subset=["uhi_tmin",D,"lp"])
    if len(s)>=20: m=cl(s,f"uhi_tmin ~ {D} + lp"); zr.append([k,lab,round(m.params[D],3),round(m.pvalues[D],3),len(s)])
pd.DataFrame(zr,columns=["koppen","climate","density_slope_ctrl_size","p","n"]).to_csv(OUT+"density_by_climate_zone.csv",index=False)

# ---------- J. within-city density (panel, 2000-2020 observation era) ----------
if HAVE_PANEL:
    p=pd.read_csv(IN+"hist_stage3_panel.csv").dropna(subset=["uhi_obs","ln_popdensity"])
    km=pr[["CityID","koppen"]].drop_duplicates("CityID"); inc=rep[["CityID","income"]].drop_duplicates("CityID")
    p=p.merge(km,on="CityID",how="left").merge(inc,on="CityID",how="left"); p["kg"]=p.koppen.astype(str).str[0]
    p=p[p.year>=2000].copy(); n=p.groupby("CityID").year.transform("count"); p=p[n>=3].copy()
    def fe(df,xc):
        dd=df.dropna(subset=["uhi_obs","ln_popdensity"]+xc).set_index(["CityID","year"])
        m=PanelOLS(dd["uhi_obs"],dd[["ln_popdensity"]+xc],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type="clustered",cluster_entity=True)
        return (round(m.params["ln_popdensity"],3),round(m.std_errors["ln_popdensity"],3),
                round(m.pvalues["ln_popdensity"],4),int(m.nobs),int(dd.index.get_level_values(0).nunique()))
    rows=[]
    for lbl,xc in [("M0 density only",[]),("M1 +GDP linear",["ln_gdp_c"]),("M2 +built",["ln_gdp_c","frac_built"]),("M3 +GDP spline",["g1","g2","g3","frac_built"])]:
        b,se,pv,nn,nc=fe(p,xc); rows.append(["incremental",lbl,b,se,pv,nn,nc])
    for g_ in ["Upper-middle","High"]:
        try:
            b,se,pv,nn,nc=fe(p[p.income==g_],[]); rows.append(["phase",g_,b,se,pv,nn,nc])
        except Exception: pass
    try:
        b,se,pv,nn,nc=fe(p[p.income.notna() & (p.income!="High")],[])
        rows.append(["phase","non-high-income",b,se,pv,nn,nc])
    except Exception: pass
    for k,lab in [("B","arid"),("C","temperate"),("D","continental")]:
        sub=p[p.kg==k]
        if sub.CityID.nunique()>=15:
            try:
                b,se,pv,nn,nc=fe(sub,[]); rows.append(["zone",f"{k} {lab}",b,se,pv,nn,nc])
            except Exception: pass
    pd.DataFrame(rows,columns=["test","group","density_slope","SE","p","n_obs","n_cities"]).to_csv(OUT+"density_within_city_sensitivity.csv",index=False)

# ---------- K. printed diagnostics: window sweep, coverage-by-year ----------
print("=== SIZE-LAW fits (per log10 pop) ==="); print(pd.read_csv(OUT+"oke_size_law_fits.csv").to_string(index=False))
if HAVE_PANEL:
    print("\n=== start-year window sweep (size between vs density within), same panel ===")
    def size_between(sub):
        cs=sub.groupby("CityID").agg(u=("uhi_obs","mean"),lp=("lp","first"),c=("country","first")).reset_index().dropna()
        m=smf.ols("u ~ lp",data=cs).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(cs.c)[0]}); return round(m.params["lp"],3),round(m.pvalues["lp"],3)
    pfull=pd.read_csv(IN+"hist_stage3_panel.csv").dropna(subset=["uhi_obs","ln_popdensity"]).merge(rep[["CityID","pop"]],on="CityID",how="left").merge(pr[["CityID","country"]],on="CityID",how="left")
    pfull=pfull[pfull["pop"]>0].copy(); pfull["lp"]=np.log10(pfull["pop"])
    for lo in [1975,1990,1995,2000,2005]:
        sb,sp=size_between(pfull[pfull.year>=lo])
        dd=pfull[pfull.year>=lo]; nn=dd.groupby("CityID").year.transform("count"); dd=dd[nn>=2].set_index(["CityID","year"])
        dm=PanelOLS(dd["uhi_obs"],dd[["ln_popdensity"]],entity_effects=True,time_effects=True,drop_absorbed=True).fit(cov_type="clustered",cluster_entity=True)
        print(f"  {lo}-2020: SIZE {sb:+.3f}(p{sp:.3f})  DENSITY {dm.params['ln_popdensity']:+.3f}(p{dm.pvalues['ln_popdensity']:.3f})")
a=pd.read_csv(IN+"annual_tavg.csv"); print("\n=== coverage by year (reporting stations) ===")
print(a.groupby("year").id.nunique().reindex([1990,1995,2000,2005,2010,2015,2020]).to_string())
print("\nDONE — all result CSVs regenerated in", OUT)
