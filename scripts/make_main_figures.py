#!/usr/bin/env python3
"""
Publication figures, drawn to the house style in figstyle.py.

  Fig. 1  which driver acts in which dimension, and with what sign  (§4.1)
  Fig. 2  the population-size law and what survives an extreme-bounds sweep (§4.2)
  Fig. 3  income enters nonlinearly, and its cross-city grip has tightened  (§4.3)
  Fig. 4  the UHI is a warm-season phenomenon; arid winters reverse it      (§4.4)
  Fig. 5  where the sample is, and what it measures there                   (§3)
  Fig. 6  published size-law coefficients on a common footing               (§5.2)
  Fig. S1 UHI level by continent, income, climate zone and country           (SI)

Both read only deposited inputs and result tables; no estimation happens here.
Titles are deliberately omitted — the caption carries the message.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN, DATA, FIG = ROOT / "data/inputs", ROOT / "data", ROOT / "figures"
fs.use()


# ---------------------------------------------------------------- Fig. 1
def figure_structure(out=FIG / "Fig1.png"):
    pr = pd.read_csv(IN / "city_uhi_predictors.csv")
    rep = pd.read_csv(IN / "representativeness.csv")[["CityID", "pop"]]
    b = pr.dropna(subset=["uhi_tmin", "ln_popdensity"]).merge(rep, on="CityID", how="left")
    b = b[b["pop"] > 0].copy()
    b["ln_size"] = np.log(b["pop"])
    z = lambda v: (v - v.mean()) / v.std()
    b["z_size"], b["z_dens"] = z(b.ln_size), z(b.ln_popdensity)

    pan = pd.read_csv(IN / "within_city_panel.csv").dropna(subset=["uhi_obs", "ln_popdensity"])
    pan = pan[pan.year >= 2000]
    # No epoch-count filter: a single-epoch city is fully absorbed by its own fixed effect and
    # mathematically inert to a two-way FE slope, but Table 1's own two-way FE estimator (and
    # PanelOLS's n_cities convention generally) still counts it, so filtering it out here would
    # only shrink the reported n without moving the slope. An earlier version of this figure
    # required >=3 epochs, which does drop informative cities and measurably shifts the slope away
    # from Table 1's own value.

    # Panel (b) shows the primary-windowe estimator rather than an illustrative stand-in. Sweeping out
    # city AND year effects by alternating projections (Frisch-Waugh-Lovell) leaves residuals
    # whose slope is the two-way fixed-effects coefficient itself, so the number on the figure
    # is the number in Table 1. Demeaning on the city alone, which leaves the common year
    # shocks in, reads about +0.35 and is not the paper's estimate.
    def sweep_out(col):
        v = pan[col].to_numpy(float).copy()
        for _ in range(60):
            v -= pd.Series(v).groupby(pan.CityID.values).transform("mean").to_numpy()
            v -= pd.Series(v).groupby(pan.year.values).transform("mean").to_numpy()
        return v
    w = pan.assign(y=sweep_out("uhi_obs"), x=sweep_out("ln_popdensity"))

    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, fs.W2 * 0.42))
    plt.subplots_adjust(wspace=.26, left=.07, right=.985, bottom=.15, top=.93)

    # (a) between cities — two drivers, opposite signs
    a = ax[0]
    for col, c, ls, lab in [("z_size", fs.BLUE, "-", "city size"),
                            ("z_dens", fs.ORANGE, "--", "population density")]:
        a.scatter(b[col], b.uhi_tmin, s=4.5, alpha=.13, color=c, lw=0, rasterized=True)
        sl, ic = np.polyfit(b[col], b.uhi_tmin, 1)
        xs = np.linspace(-2.1, 2.1, 40)
        a.plot(xs, sl * xs + ic, color=c, lw=1.9, ls=ls, zorder=4)
        # direct label inside the axes, at the end of its own line
        ytxt = sl * 1.55 + ic
        a.text(1.62, ytxt + (0.34 if sl > 0 else -0.34), f"{lab}\n{sl:+.2f} °C per SD",
               color=c, fontsize=7.5, ha="center",
               va="bottom" if sl > 0 else "top", fontweight="bold", linespacing=1.3,
               bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))
    a.axhline(0, color=fs.GREY, lw=.5, zorder=1)
    a.set_xlim(-2.3, 2.3); a.set_ylim(-2.6, 3.9)
    a.set_xlabel("city attribute (standard deviations from the mean)")
    a.set_ylabel("nocturnal UHI (°C)")
    fs.panel_label(a, "a")
    fs.annotate(a, .02, .97, f"{len(b)} cities", color=fs.GREY)

    # (b) within cities
    a = ax[1]
    a.scatter(w.x, w.y, s=4.5, alpha=.16, color=fs.ORANGE, lw=0, rasterized=True)
    s2, i2 = np.polyfit(w.x, w.y, 1)
    xs = np.linspace(w.x.quantile(.01), w.x.quantile(.99), 40)
    a.plot(xs, s2 * xs + i2, color=fs.ORANGE, lw=1.9, zorder=4)
    a.axhline(0, color=fs.GREY, lw=.5, zorder=1)
    a.set_xlim(w.x.quantile(.01), w.x.quantile(.99))
    a.set_ylim(w.y.quantile(.02), w.y.quantile(.98))
    a.set_xlabel("population density (ln), city and year effects swept out")
    a.set_ylabel("UHI (°C), city and year effects swept out")
    fs.panel_label(a, "b")
    fs.annotate(a, .02, .97,
                f"{s2:+.2f} °C per ln unit (two-way fixed effects)\n"
                f"{w.CityID.nunique()} cities, {len(w):,} city-epochs, 2000–2020",
                color=fs.ORANGE)
    fig.savefig(out)
    plt.close(fig)
    ss = np.polyfit(b.z_size, b.uhi_tmin, 1)[0]; sd = np.polyfit(b.z_dens, b.uhi_tmin, 1)[0]
    print(f"  Fig 1  between size {ss:+.2f} / density {sd:+.2f} per SD (n={len(b)}); "
          f"within {s2:+.2f} ({w.CityID.nunique()} cities)")


# ---------------------------------------------------------------- Fig. 6
def figure_recalibration(out=FIG / "Fig6.png"):
    d = pd.read_csv(DATA / "oke_literature_matched_coverage.csv")
    ours = d[d.source.str.startswith("OURS")]
    lit = d[~d.source.str.startswith("OURS")]

    rows = []
    for _, r in lit.iterrows():
        rows.append((r.source.split(" (")[0] + " — " + r.region,
                     r.slope_per_tenfold_ourform, r.uhi_time, "lit"))
    for reg in ["Global (all)", "Europe", "North America", "United States"]:
        for el in ["day", "mean", "night"]:
            m = ours[(ours.region == reg) & (ours.uhi_time == el)]
            if len(m):
                rows.append((f"This study — {reg.replace(' (all)','')}",
                             float(m.slope_per_tenfold_ourform.iloc[0]), el, "ours"))
    df = pd.DataFrame(rows, columns=["label", "slope", "element", "kind"])
    order = list(dict.fromkeys(df.label))
    ypos = {l: i for i, l in enumerate(order)}

    style = {"day": ("v", fs.BLUE), "mean": ("o", fs.GREY), "night": ("^", fs.ORANGE),
             "MAX (nocturnal)": ("*", fs.ORANGE), "annual avg": ("s", fs.GREY)}
    fig, ax = plt.subplots(figsize=(fs.W2 * .80, fs.W2 * .40))
    plt.subplots_adjust(left=.26, right=.985, bottom=.17, top=.97)
    ax.axvline(0, color=fs.GREY, lw=.6, zorder=1)
    for lab, yy in ypos.items():
        sub = df[df.label == lab]
        ax.plot([sub.slope.min(), sub.slope.max()], [yy, yy], color=fs.GREY,
                lw=.6, alpha=.5, zorder=2)
        for _, r in sub.iterrows():
            mk, c = style.get(r.element, ("o", fs.GREY))
            ax.scatter(r.slope, yy, marker=mk, s=120 if mk == "*" else 34, color=c,
                       zorder=4, edgecolor="white", lw=.5)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()))
    ax.set_xlabel("size-law slope (°C per tenfold population), common functional form")
    ax.set_ylim(-.7, len(order) - .3)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], marker=m, ls="", color=c, label=l, ms=5.5)
         for l, (m, c) in [("daytime", style["day"]), ("annual mean", style["mean"]),
                           ("nocturnal", style["night"]),
                           ("calm-clear maximum", style["MAX (nocturnal)"])]]
    ax.legend(handles=h, loc="lower right", ncol=2, borderaxespad=.6, columnspacing=1.4)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Fig 6  {len(df)} estimates, {len(order)} sources")




# ---------------------------------------------------------------- Fig. 2
def figure_sizelaw(out=FIG / "Fig2.png"):
    """The size law: the relationship, its element-dependence, its zone stability,
    its insensitivity to station siting, and which candidate drivers survive an
    extreme-bounds sweep. Panels a–e, all from deposited tables."""
    pr = pd.read_csv(IN / "city_uhi_predictors.csv")
    rep = pd.read_csv(IN / "representativeness.csv")[["CityID", "pop"]]
    c = pr.merge(rep, on="CityID", how="left")
    c = c[c["pop"] > 0].copy()
    c["lp"] = np.log10(c["pop"])
    c["uhi_max"] = 2 * c.uhi_tavg - c.uhi_tmin
    fits = pd.read_csv(DATA / "oke_size_law_fits.csv")
    zones = pd.read_csv(DATA / "oke_size_law_by_climate_zone.csv")
    dist = pd.read_csv(DATA / "oke_station_distance_sensitivity.csv")

    eba = pd.read_csv(DATA / "eba_uhi_level_tmin.csv")   # nocturnal, matches the rest of this figure

    fig = plt.figure(figsize=(fs.W2, fs.W2 * 0.70))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 1, 1], hspace=.50, wspace=.46,
                          left=.062, right=.975, top=.945, bottom=.085)
    ELEM = {"nighttime (TMIN)": ("uhi_tmin", fs.ORANGE, "nocturnal", "-"),
            "mean (TAVG)": ("uhi_tavg", fs.GREY, "annual mean", "--"),
            "daytime (TMAX)": ("uhi_max", fs.BLUE, "daytime", ":")}

    # (a) the relationship
    a = fig.add_subplot(gs[:, 0])
    a.scatter(c.lp, c.uhi_tmin, s=4, alpha=.14, color=fs.ORANGE, lw=0, rasterized=True)
    xs = np.linspace(c.lp.quantile(.01), c.lp.quantile(.99), 40)
    for _, r in fits[fits.uhi_measure.isin(ELEM)].iterrows():   # skip the common-sample row
        col, colr, lab, ls = ELEM[r.uhi_measure]
        a.plot(xs, r.intercept + r.slope_per_log10pop * xs, color=colr, lw=1.9,
               ls=ls, zorder=4)
        a.text(xs[-1] + .05, r.intercept + r.slope_per_log10pop * xs[-1],
               f"{lab}\n{r.slope_per_log10pop:+.2f}", color=colr, fontsize=7,
               va="center", fontweight="bold", linespacing=1.25)
    a.axhline(0, color=fs.GREY, lw=.5)
    a.set_xlim(c.lp.quantile(.01), c.lp.quantile(.99) + .95)
    a.set_ylim(-2.6, 3.9)
    a.set_xlabel("city population (log₁₀)"); a.set_ylabel("UHI (°C)")
    fs.panel_label(a, "a", dx=-.14)

    # (b) element-specific slopes
    b = fig.add_subplot(gs[0, 1])
    lab_order = ["daytime (TMAX)", "mean (TAVG)", "nighttime (TMIN)"]
    for i, k in enumerate(lab_order):
        r = fits[fits.uhi_measure == k].iloc[0]
        _, colr, lab, _ls = ELEM[k]
        b.errorbar(i, r.slope_per_log10pop, yerr=1.96 * r.slope_SE, fmt="o", ms=4.5,
                   color=colr, capsize=2.2, lw=1.2)
    b.axhline(0, color=fs.GREY, lw=.5)
    b.set_xticks(range(3)); b.set_xticklabels(["day", "mean", "night"])
    b.set_ylabel("slope (°C per\ntenfold population)")
    b.set_xlim(-.5, 2.5)
    fs.panel_label(b, "b", dx=-.30)

    # (c) by climate zone
    cc = fig.add_subplot(gs[0, 2])
    zz = zones.dropna(subset=["night_size_slope"])
    for i, (_, r) in enumerate(zz.iterrows()):
        cc.errorbar(i, r.night_size_slope, yerr=1.96 * r.SE, fmt="s", ms=4,
                    color=fs.ORANGE, capsize=2.2, lw=1.2)
        cc.annotate(f"n={int(r.n)}", (i, r.night_size_slope), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.3, color=fs.GREY)
    cc.axhline(0, color=fs.GREY, lw=.5)
    cc.set_xticks(range(len(zz)))
    cc.set_xticklabels([str(x) for x in zz.climate], fontsize=6.6, rotation=30, ha="right")
    cc.set_ylabel("nocturnal slope")
    cc.set_xlim(-.5, len(zz) - .5)
    fs.panel_label(cc, "c", dx=-.30)

    # (d) siting: distant urban stations read a smaller UHI, but the slope is unchanged
    dd = fig.add_subplot(gs[1, 1])
    bins = dist[dist.test.str.contains("station <=5km|5-15km|>15km")].copy()
    lbl = ["≤5", "5–15", ">15"]
    x = np.arange(len(bins))
    dd.bar(x - .19, bins.mean_uhi, .36, color=fs.ORANGE, alpha=.85)
    dd.bar(x + .19, bins.estimate, .36, color=fs.BLUE, alpha=.85)
    for xi, (u, sl) in enumerate(zip(bins.mean_uhi, bins.estimate)):
        dd.text(xi - .19, u + .02, f"{u:.2f}", ha="center", fontsize=6.2, color=fs.ORANGE)
        dd.text(xi + .19, sl + .02, f"{sl:.2f}", ha="center", fontsize=6.2, color=fs.BLUE)
    dd.set_xticks(x); dd.set_xticklabels(lbl)
    dd.set_xlabel("urban station to centre (km)")
    dd.set_ylabel("°C  /  °C per tenfold")
    dd.set_ylim(0, 1.32)
    fs.annotate(dd, .0, 1.0, "mean UHI", color=fs.ORANGE, size=6.8)
    fs.annotate(dd, .0, .90, "size-law slope", color=fs.BLUE, size=6.8)
    fs.annotate(dd, .98, 1.0, "interaction\np = 0.55", color=fs.GREY, ha="right", size=6.5)
    fs.panel_label(dd, "d", dx=-.30)

    # (e) extreme bounds: which candidate drivers survive, of the up to 384 specs each appears in
    ee = fig.add_subplot(gs[1, 2])
    SHORT = {"City size": "city size", "Pop density": "density", "Wind": "wind",
             "Veg contrast": "veg contrast", "GDP/capita": "GDP/capita",
             "|Latitude|": "abs. latitude", "Built-up": "built-up",
             "Coastal": "coast dist."}
    short = lambda d: next(v for k, v in SHORT.items() if d.startswith(k))
    eb = eba.sort_values("pct_sig")            # bottom-to-top = weakest first
    ys = np.arange(len(eb))
    for y, (_, r) in zip(ys, eb.iterrows()):
        colr = (fs.BLUE if "City size" in r.driver else
                fs.ORANGE if "density" in r.driver else fs.GREY)
        solid = r.pct_sig >= 95
        ee.plot([r.lo, r.hi], [y, y], color=colr, lw=1.5,
                solid_capstyle="butt", alpha=.9 if solid else .55)
        ee.plot(r.median_beta, y, "o", ms=4.6 if solid else 3.6, color=colr,
                mfc=colr if solid else "white", mew=1.1, zorder=4)
        ee.text(1.02, y, f"{r.pct_sig:g}%", transform=ee.get_yaxis_transform(),
                va="center", fontsize=6.6,
                color=fs.INK if solid else fs.GREY,
                fontweight="bold" if solid else "normal")
    ee.axvline(0, color=fs.INK, lw=.6)
    ee.set_yticks(ys); ee.set_yticklabels([short(d) for d in eb.driver], fontsize=6.8)
    ee.set_ylim(-.7, len(eb) - .3)
    ee.set_xlim(-.42, .42)
    ee.set_xticks([-.4, -.2, 0, .2, .4])
    ee.set_xlabel("β per SD (°C)")
    ee.spines["left"].set_visible(False); ee.tick_params(axis="y", length=0)
    fs.annotate(ee, 1.02, 1.10, "% of specs\np < 0.05", color=fs.GREY, size=6.3, ha="left")
    fs.panel_label(ee, "e", dx=-.42)

    fig.savefig(out); plt.close(fig)
    print(f"  Fig 2  panels a–e rebuilt ({len(c)} cities; "
          f"panel e nocturnal extreme bounds, city size {int(eba.n_specs.max())} of 384 specs)")


# ---------------------------------------------------------------- Fig. 5
def figure_geography(out=FIG / "Fig5.png"):
    """Where the sample is, and what it measures there — the scope within which the
    scaling relationships hold. Basemap: Natural Earth 110 m land (public domain)."""
    import geopandas as gpd
    land = gpd.read_file(HERE / "assets/ne_110m_land.shp")
    pr = pd.read_csv(IN / "city_uhi_predictors.csv")
    match = pd.read_csv(IN / "city_station_match.csv")[["city_id", "lat", "lon"]]
    d = pr.merge(match, on="city_id", how="inner").dropna(subset=["uhi_tavg", "lat", "lon"])

    fig, ax = plt.subplots(figsize=(fs.W2, fs.W2 * 0.46))
    plt.subplots_adjust(left=.01, right=.90, top=.99, bottom=.06)
    land.plot(ax=ax, color="#EFEFEF", edgecolor="#D8D8D8", linewidth=.3, zorder=1)

    v = 2.0
    sc = ax.scatter(d.lon, d.lat, c=d.uhi_tavg.clip(-v, v), cmap="RdBu_r",
                    vmin=-v, vmax=v, s=7, lw=.15, edgecolor="white", zorder=3)
    ax.set_xlim(-170, 180); ax.set_ylim(-58, 80)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    cax = fig.add_axes([.915, .30, .012, .42])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("city-mean UHI (°C)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(.4)

    counts = d.continent.value_counts()
    keep = [c for c in ["North America", "Europe", "Asia", "Oceania"] if c in counts.index]
    fs.annotate(ax, .012, .34,
                "\n".join(f"{c}  {counts[c]}" for c in keep) +
                f"\n\ntotal  {len(d)} cities\nnone in Africa or South America",
                color=fs.INK, size=7)
    fig.savefig(out); plt.close(fig)
    print(f"  Fig 5  {len(d)} cities mapped; {d.continent.value_counts().to_dict()}")


# ---------------------------------------------------------------- Fig. 3
def figure_income(out=FIG / "Fig3.png"):
    """Income enters the UHI nonlinearly, and its cross-city grip has tightened.
    Panels a-b from gdp_rcs.py; panels c-d from income_over_time.py."""
    cur = pd.read_csv(DATA / "gdp_rcs_partial_effect.csv")
    dist = pd.read_csv(DATA / "gdp_distribution.csv").ln_gdp.dropna()
    grad = pd.read_csv(DATA / "income_gradient_over_time.csv").sort_values("year")
    traj = pd.read_csv(DATA / "income_tercile_trajectory.csv")

    fig = plt.figure(figsize=(fs.W2, fs.W2 * 0.66))
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.26,
                          left=.075, right=.985, top=.94, bottom=.09)

    # (a, b) the spline shape, one element each
    for j, (dv, colr, lab, letter, ylim) in enumerate(
            [("mean (TAVG)", fs.SLATE, "annual-mean UHI", "a", (-1.32, .95)),
             ("night (TMIN)", fs.ORANGE, "nocturnal UHI", "b", (-1.02, 1.05))]):
        ax = fig.add_subplot(gs[0, j])
        s = cur[cur.dv == dv]
        rug = ax.twinx()
        rug.hist(dist[(dist >= s.ln_gdp.min()) & (dist <= s.ln_gdp.max())],
                 bins=34, color="#E9E9E9", zorder=0)
        rug.set_yticks([]); rug.set_ylim(0, dist.size * .55)
        for sp in rug.spines.values():
            sp.set_visible(False)
        ax.set_zorder(rug.get_zorder() + 1); ax.patch.set_visible(False)

        ax.fill_between(s.ln_gdp, s.lo, s.hi, color=colr, alpha=.16, lw=0, zorder=2)
        ax.plot(s.ln_gdp, s.effect, color=colr, lw=2.0, zorder=4)
        ax.axhline(0, color=fs.GREY, lw=.5, zorder=1)
        ax.set_xlabel("city income (ln GDP per capita)")
        ax.set_ylabel(f"{lab}, relative to a\nmean-income city (°C)")
        ax.set_ylim(*ylim)
        fs.annotate(ax, .03, .97, f"{lab}\n{int(s.n.iloc[0])} cities", color=colr)
        fs.annotate(ax, .97, .06, "grey: distribution\nof city incomes",
                    color=fs.GREY, ha="right", va="bottom", size=6.6)
        fs.panel_label(ax, letter, dx=-.20)

    # (c) the cross-city gradient, snapshot by snapshot
    c = fig.add_subplot(gs[1, 0])
    c.fill_between(grad.year, grad.lo, grad.hi, color=fs.PURPLE, alpha=.15, lw=0)
    c.plot(grad.year, grad.gradient, "-o", color=fs.PURPLE, lw=1.9, ms=4.4, zorder=4)
    c.axhline(0, color=fs.GREY, lw=.5)
    c.set_xticks(grad.year)
    c.set_xlabel("five-year snapshot")
    c.set_ylabel("income–UHI gradient\n(°C per unit ln GDP per capita)")
    c.set_ylim(-.05, .72)
    pk = grad.loc[grad.gradient.idxmax()]
    c.annotate(f"{pk.gradient:+.2f}", (pk.year, pk.gradient), textcoords="offset points",
               xytext=(2, 7), fontsize=7, color=fs.PURPLE, fontweight="bold")
    fs.annotate(c, .03, .97, f"{grad.gradient.iloc[0]:+.2f} to {pk.gradient:+.2f} °C\n"
                             "size and climate held fixed", color=fs.PURPLE)
    fs.panel_label(c, "c", dx=-.20)

    # (d) the terciles that gradient is made of
    d = fig.add_subplot(gs[1, 1])
    STY = {"higher-GDP": (fs.PURPLE, "-", "highest-income third"),
           "lower-GDP": (fs.BLUE, "--", "lowest-income third"),
           "mid-GDP": (fs.GREY, ":", "middle third")}
    for t, (colr, ls, lab) in STY.items():
        s = traj[traj.tercile == t].sort_values("year")
        d.plot(s.year, s.median_uhi, ls, color=colr, lw=1.8, marker="o", ms=3.6, zorder=4)
        d.text(s.year.iloc[-1] + .7, s.median_uhi.iloc[-1], lab, color=colr,
               fontsize=7, va="center", fontweight="bold")
    d.set_xticks(sorted(traj.year.unique()))
    d.set_xlim(1999, 2029)
    d.set_xlabel("five-year snapshot")
    d.set_ylabel("median UHI (°C)")
    d.set_ylim(0, .48)
    fs.annotate(d, .03, .97, "cities held in their 2020 income tercile", color=fs.GREY)
    fs.panel_label(d, "d", dx=-.20)

    fig.savefig(out); plt.close(fig)
    print(f"  Fig 3  gradient {grad.gradient.iloc[0]:+.2f}→{grad.gradient.max():+.2f}; "
          f"RCS on {int(cur[cur.dv=='mean (TAVG)'].n.iloc[0])}/"
          f"{int(cur[cur.dv=='night (TMIN)'].n.iloc[0])} cities")


# ---------------------------------------------------------------- Fig. 4
def figure_seasonal(out=FIG / "Fig4.png"):
    """The UHI is a warm-season phenomenon, and in arid cities the cold season
    reverses it outright. From seasonal_uhi.py."""
    z = pd.read_csv(DATA / "seasonal_uhi_by_zone.csv")
    z = z[z.split == "koppen"].set_index("group").loc[
        ["A tropical", "B arid", "C temperate", "D continental"]].reset_index()

    fig, ax = plt.subplots(figsize=(fs.W2 * .74, fs.W2 * .40))
    plt.subplots_adjust(left=.10, right=.985, top=.90, bottom=.14)

    x = np.arange(len(z))
    ax.bar(x - .19, z.warm, .36, color=fs.ORANGE, alpha=.9, lw=0)
    ax.bar(x + .19, z.cold, .36, color=fs.BLUE, alpha=.9, lw=0)
    for xi, r in zip(x, z.itertuples()):
        ax.text(xi - .19, r.warm + .022, f"{r.warm:+.2f}", ha="center",
                fontsize=7, color=fs.ORANGE, fontweight="bold")
        ax.text(xi + .19, r.cold + (.022 if r.cold >= 0 else -.055), f"{r.cold:+.2f}",
                ha="center", fontsize=7, color=fs.BLUE, fontweight="bold")
        ax.text(xi, -.365, f"n = {int(r.n)}", ha="center", fontsize=6.8, color=fs.GREY)
    ax.axhline(0, color=fs.INK, lw=.7)

    ax.set_xticks(x)
    ax.set_xticklabels([g.split(" ", 1)[1] for g in z.group])
    ax.set_xlim(-.6, len(z) - .4)
    ax.set_ylim(-.42, .88)
    ax.set_ylabel("median UHI (°C)")
    ax.set_xlabel("Köppen climate zone")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5)

    fs.annotate(ax, .015, .98, "warm season", color=fs.ORANGE, size=8)
    fs.annotate(ax, .015, .90, "cold season", color=fs.BLUE, size=8)
    ax.text(2.55, -.155, "arid winters are urban cool islands,\nnot heat islands — the only zone\nwhere the sign reverses",
            fontsize=7, color=fs.BLUE, ha="center", va="center", linespacing=1.4)
    fig.savefig(out); plt.close(fig)
    print(f"  Fig 4  {int(z.n.sum())} cities across {len(z)} zones; "
          f"arid {z.loc[1,'warm']:+.2f} warm / {z.loc[1,'cold']:+.2f} cold")




# ---------------------------------------------------------------- Fig. S1
def figure_groupings(out=FIG / "FigS1.png"):
    """Typical UHI by continent, income group, climate zone and country, for the
    annual mean and the night. Medians with 95% bootstrap intervals; from groupings.py."""
    g = pd.read_csv(DATA / "uhi_by_grouping.csv")
    XLO, XHI = -1.25, 1.55

    fig = plt.figure(figsize=(fs.W2, fs.W2 * 0.72))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[4, 3, 4],
                          hspace=.55, wspace=.52,
                          left=.135, right=.90, top=.905, bottom=.075)

    def panel(ax, split, letter, title, last=False, dy=1.16, dx=-.42):
        s = g[g.split == split].iloc[::-1].reset_index(drop=True)   # first group on top
        ys = np.arange(len(s))
        for y, r in zip(ys, s.itertuples()):
            for lo, hi, mid, colr, off, ms in [
                    (r.mean_lo, r.mean_hi, r.mean_uhi, fs.SLATE, .16, 4.2),
                    (r.night_lo, r.night_hi, r.night_uhi, fs.ORANGE, -.16, 4.2)]:
                if not np.isfinite(mid):
                    if colr == fs.ORANGE:
                        ax.text(XHI - .04, y + off, "no night data", ha="right",
                                va="center", fontsize=5.8, color=fs.GREY, style="italic")
                    continue
                # intervals wider than the axis are drawn to the edge and capped
                l, h = max(lo, XLO), min(hi, XHI)
                ax.plot([l, h], [y + off] * 2, color=colr, lw=1.2, alpha=.85,
                        solid_capstyle="butt", zorder=3)
                for v, edge, mk in [(lo, XLO, "<"), (hi, XHI, ">")]:
                    if (v < XLO) if mk == "<" else (v > XHI):
                        ax.plot(edge, y + off, mk, color=colr, ms=3, zorder=3)
                ax.plot(mid, y + off, "o", color=colr, ms=ms, mec="white", mew=.6, zorder=5)
            ax.text(1.015, y, f"{int(r.n)}", transform=ax.get_yaxis_transform(),
                    fontsize=6.5, va="center", color=fs.GREY)
            ax.text(1.145, y, f"{r.trend:+.2f}", transform=ax.get_yaxis_transform(),
                    fontsize=6.5, va="center", color=fs.GREY)
        ax.axvline(0, color=fs.INK, lw=.6, zorder=1)
        ax.set_yticks(ys); ax.set_yticklabels(s.group, fontsize=7.2)
        ax.set_ylim(-.65, len(s) - .35)
        ax.set_xlim(XLO, XHI)
        ax.set_xticks([-1, -.5, 0, .5, 1, 1.5])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.text(1.015, len(s) - .25, "n", transform=ax.get_yaxis_transform(),
                fontsize=6.3, va="center", color=fs.GREY)
        ax.text(1.145, len(s) - .25, "trend", transform=ax.get_yaxis_transform(),
                fontsize=6.3, va="center", color=fs.GREY)
        fs.annotate(ax, .0, dy - .03, title, color=fs.INK, size=7.6)
        fs.panel_label(ax, letter, dx=dx, dy=dy)
        if last:
            ax.set_xlabel("median UHI, 2001–2020 (°C)")
        else:
            ax.tick_params(axis="x", labelbottom=True)
        return s

    panel(fig.add_subplot(gs[0, 0]), "continent", "a", "by continent")
    panel(fig.add_subplot(gs[1, 0]), "income", "b", "by World Bank income group")
    panel(fig.add_subplot(gs[2, 0]), "koppen", "c", "by Köppen climate zone", last=True)
    panel(fig.add_subplot(gs[:, 1]), "country", "d", "by country (≥15 cities)",
          last=True, dy=1.045, dx=-.16)

    fig.text(.135, .982, "annual mean", color=fs.SLATE, fontsize=8, fontweight="bold")
    fig.text(.245, .982, "night", color=fs.ORANGE, fontsize=8, fontweight="bold")
    fig.text(.135, .958, "median city with a 95% bootstrap interval; arrows mark intervals "
                         "running past the axis; trend is °C per decade over the same window",
             color=fs.GREY, fontsize=6.8)
    fig.savefig(out); plt.close(fig)
    print(f"  Fig S1  {len(g)} groups; night missing for "
          f"{int(g.night_uhi.isna().sum())} (too few TMIN cities)")


if __name__ == "__main__":
    figure_structure()
    figure_sizelaw()
    figure_income()
    figure_seasonal()
    figure_geography()
    figure_recalibration()
    figure_groupings()
