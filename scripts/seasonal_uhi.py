#!/usr/bin/env python3
"""
Is the urban heat island a warm-season or a cold-season phenomenon?  (Fig. 4)

For each city the UHI is computed twice, once from June-August station means and
once from December-February means, and the two are then relabelled by hemisphere
so that "warm" and "cold" mean the city's own summer and winter rather than a
fixed pair of months. Everything else follows the paper's standard construction:
urban minus the median of at least three rural references, a 6.5 C/km lapse-rate
correction for the urban-rural elevation difference, the 12 km contamination
screen on rural candidates, and the 2001-2020 window.

The seasonal question matters because a mechanism that operates through daytime
storage and nocturnal release should be strongest in summer, whereas one
operating through space heating should be strongest in winter. Splitting by
climate zone separates these: the arid zone is the case where the sign actually
reverses.

Provenance
----------
The city-level product ``seasonal_uhi_cities.csv`` ships with this package. It
is derived from seasonal station means (``seasonal_tavg.csv``), which are in
turn parsed from the raw GHCN-Daily ``.dly`` archive -- roughly 11 GB that is
not redistributed here, since it is available unchanged from NOAA. Rebuilding
from the archive is therefore a two-step job; the per-city output is deposited
so that Fig. 4 and the numbers in Section 4 can be checked without it.

Input :  data/inputs/seasonal_uhi_cities.csv
Output:  data/seasonal_uhi_by_zone.csv     (drives Fig. 4)
         plus the continent and latitude-band splits reported in the text
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/inputs/seasonal_uhi_cities.csv"
OUT = ROOT / "data/seasonal_uhi_by_zone.csv"

ZONES = ["A tropical", "B arid", "C temperate", "D continental"]
CONTINENTS = ["North America", "Europe", "Asia", "Oceania"]
BANDS = ["low (<35°)", "mid (35-50°)", "high (>50°)"]
MIN_N = 10


def summarise(d, by, keys):
    rows = []
    for k in keys:
        s = d[d[by] == k]
        if len(s) < MIN_N:
            continue
        rows.append(dict(split=by, group=k, n=len(s),
                         warm=round(float(s.warm.median()), 4),
                         cold=round(float(s.cold.median()), 4),
                         warm_minus_cold=round(float(s.seasonal_diff.median()), 4),
                         pct_warm_stronger=round(100 * float((s.seasonal_diff > 0).mean()))))
    return rows


def main():
    d = pd.read_csv(SRC)
    print(f"{len(d)} cities")
    print(f"warm-season UHI {d.warm.median():+.2f} °C | cold-season {d.cold.median():+.2f} °C | "
          f"warm-cold {d.seasonal_diff.median():+.2f} °C")
    print(f"{100 * np.mean(d.seasonal_diff > 0):.0f}% of cities have the stronger UHI in their warm season")

    rows = (summarise(d, "koppen", ZONES)
            + summarise(d, "continent", CONTINENTS)
            + summarise(d, "latb", BANDS))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print()
    print(out.to_string(index=False))

    arid = out[(out.split == "koppen") & (out.group == "B arid")].iloc[0]
    print(f"\nthe reversal: arid cities are {arid.warm:+.2f} °C in summer and "
          f"{arid.cold:+.2f} °C in winter — an urban cool island, not a heat island")


if __name__ == "__main__":
    main()
