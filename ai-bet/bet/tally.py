"""HISTORIAL: como correram os boletins e se os agentes ajudam.

  - acerto e lucro "a 1 unidade" por aposta (como um tipster mostraria);
  - CLV contra o fecho da Pinnacle, quando existe (ligas do football-data);
  - log loss das camadas mercado / modelos / agentes nos jogos já disputados.
"""
from __future__ import annotations

import math

import numpy as np

from . import db, metrics


def pick_stats(rows):
    s = [p for p in rows if p["status"] != "open"]
    clv = [p["clv"] for p in s if p["clv"] is not None]
    units = [p["units"] or 0 for p in s]
    wins = sum(1 for p in s if p["status"] in ("won", "half_won"))
    out = {"n": len(s), "open": sum(1 for p in rows if p["status"] == "open"), "wins": wins,
           "hit": round(wins / len(s), 3) if s else None,
           "units": round(sum(units), 2), "roi": round(sum(units) / len(s), 4) if s else None,
           "avg_odds": round(float(np.mean([p["odds"] for p in s])), 2) if s else None,
           "exp_hit": round(float(np.mean([p["p"] for p in s])), 3) if s else None,
           "clv": round(float(np.mean(clv)), 4) if clv else None, "n_clv": len(clv)}
    if len(clv) >= 5:
        sd = float(np.std(clv, ddof=1)) or 1e-9
        out["clv_z"] = round(float(np.mean(clv)) / (sd / math.sqrt(len(clv))), 2)
    return out


def refresh():
    allp = db.picks()
    singles = [p for p in allp if p["kind"] == "single"]
    multis = [p for p in allp if p["kind"] == "multi"]
    safes = [p for p in allp if p["kind"] == "safe"]
    by_market = {}
    for p in singles:
        fam = ("Handicap" if p["market"].startswith("ah") else "Golos" if p["market"].startswith("ou")
               else {"1x2": "1X2", "dc": "Dupla hipótese", "btts": "Ambas marcam"}.get(p["market"], p["market"]))
        by_market.setdefault(fam, []).append(p)
    lp = [r for r in db.layer_all() if r["res"] is not None]
    by = {}
    for r in lp:
        by.setdefault(r["mid"], {})[r["layer"]] = (r["p"], r["res"])
    common = [m for m, d in by.items() if {"mercado", "modelos", "agentes"} <= d.keys()]
    layers = {}
    for L in ("mercado", "modelos", "agentes"):
        p = np.array([by[m][L][0] for m in common]) if common else np.zeros((0, 3))
        y = np.array([by[m][L][1] for m in common], int)
        layers[L] = metrics.summary(p, y) if len(y) else {"n": 0}
    t = {"singles": pick_stats(singles), "multi": pick_stats(multis), "safe": pick_stats(safes),
         "by_market": {k: pick_stats(v) for k, v in by_market.items()},
         "by_stars": {str(k): pick_stats([p for p in singles if p["stars"] == k]) for k in (1, 2, 3)},
         "layers": layers}
    db.kv_set("tally", t)
    return t
