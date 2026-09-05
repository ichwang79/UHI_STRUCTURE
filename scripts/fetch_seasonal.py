"""Stream GHCN-Daily .dly files for the study's station list from NCEI and keep only the
per station-year seasonal means (JJA, DJF) of TMAX, TMIN and TAVG. Nothing is stored on disk
except the output table. Same quality rules as the annual build: blank QFLAG only, >=20 valid
days per month, all three months of a season present.

Output: seasonal_by_elem.csv  (id, year, jja_tmin, jja_tmax, jja_tavg, djf_tmin, djf_tmax, djf_tavg)
"""
import os, sys, time, numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
G = os.environ.get("UHI_AIR_DATA", ".")   # station list (need_broad_meta.csv / ghcnd-stations.txt) from the air record
OUT = os.path.join(os.environ.get("UHI_EXTRA_INPUTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs")), "seasonal_by_elem.csv")
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly"
MIN_DAYS = 20

def parse(text):
    mon = {}
    for ln in text.splitlines():
        el = ln[17:21]
        if el not in ("TMAX", "TMIN", "TAVG"): continue
        yr = int(ln[11:15]); mo = int(ln[15:17]); vals = []
        for d in range(31):
            b = 21 + d * 8; v = ln[b:b + 5]
            if v == "-9999" or ln[b + 6] != " ": continue
            try: vals.append(int(v) / 10.0)
            except ValueError: pass
        if len(vals) >= MIN_DAYS: mon.setdefault((yr, mo), {})[el] = np.mean(vals)
    for k, e in mon.items():
        if "TMAX" in e and "TMIN" in e: e["TAVG"] = (e["TMAX"] + e["TMIN"]) / 2
    out = []
    for yr in sorted({y for y, _ in mon}):
        r = {"year": yr}
        for el in ("TMIN", "TMAX", "TAVG"):
            j = [mon[(yr, m)][el] for m in (6, 7, 8) if (yr, m) in mon and el in mon[(yr, m)]]
            w = [mon[(yr - 1, 12)][el]] if (yr - 1, 12) in mon and el in mon[(yr - 1, 12)] else []
            w += [mon[(yr, m)][el] for m in (1, 2) if (yr, m) in mon and el in mon[(yr, m)]]
            r["jja_" + el.lower()] = np.mean(j) if len(j) == 3 else np.nan
            r["djf_" + el.lower()] = np.mean(w) if len(w) == 3 else np.nan   # DJF = Dec(y-1), Jan, Feb
        out.append(r)
    return out

def fetch(sid, tries=3):
    for t in range(tries):
        try:
            r = requests.get(URL.format(sid), timeout=120)
            if r.status_code == 200: return sid, parse(r.text), len(r.content)
            if r.status_code == 404: return sid, [], 0
        except Exception:
            time.sleep(2 * (t + 1))
    return sid, None, 0

def main():
    if os.path.exists(f"{G}/need_broad.txt"):
        ids = [l.strip()[:11] for l in open(f"{G}/need_broad.txt") if l.strip()]
    else:   # the air record ships the station list as need_broad_meta.csv
        ids = pd.read_csv(f"{G}/need_broad_meta.csv", dtype={"id": str}).id.unique().tolist()
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT, usecols=["id"], dtype={"id": str}).id.unique())
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} stations, {len(done)} already done, {len(todo)} to fetch", flush=True)
    rows, nbytes, fail, t0 = [], 0, [], time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch, s): s for s in todo}
        for k, f in enumerate(as_completed(futs), 1):
            sid, recs, nb = f.result(); nbytes += nb
            if recs is None: fail.append(sid)
            else:
                for r in recs: rows.append({"id": sid, **r})
            if k % 200 == 0 or k == len(todo):
                pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False); rows = []
                print(f"  {k}/{len(todo)}  {nbytes/1e9:.2f} GB  {time.time()-t0:.0f}s  failed {len(fail)}", flush=True)
    if fail: open(OUT + ".failed", "w").write("\n".join(fail))
    d = pd.read_csv(OUT, dtype={"id": str}); print(f"saved {OUT}: {d.id.nunique()} stations, {len(d)} station-years")

if __name__ == "__main__":
    main()
