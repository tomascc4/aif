"""Backtest walk-forward dos modelos estatísticos (sem LLM: os LLMs conhecem os
resultados passados dos dados de treino, o que inflacionava os números).

Aposta 1 unidade na melhor seleção (1X2 ou mais/menos 2.5) de cada jogo com
valor >= min_edge, contra a melhor odd de abertura. Mede acerto, ROI e CLV
contra o fecho da Pinnacle. A combinação de modelos é reaprendida de 60 em 60
dias, só com o passado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics, odds
from .quant import M1X2, QuantEngine


def run(cfg: dict, rows: list[dict], refit_days: int = 60, strategy: str = "ensemble") -> dict:
    b = cfg.get("boletim", {})
    min_edge, max_dis = b.get("min_edge", 0.03), b.get("max_disagree", 0.08)
    lo, hi = b.get("min_odds", 1.3), b.get("max_odds", 4.5)
    rows = sorted([r for r in rows if r.get("mkt") and "res" in r], key=lambda r: r["date"])
    if len(rows) < 400:
        return {"error": f"poucos jogos fora-da-amostra ({len(rows)})"}
    qe = QuantEngine(cfg)
    test_start = rows[len(rows) // 3]["date"]
    last_fit, bets, ys = None, [], []
    evals = {n: [] for n in M1X2 + ["ens"]}
    for r in rows:
        if r["date"] < test_start:
            continue
        day = r["date"][:10]
        if last_fit is None or (pd.Timestamp(day) - pd.Timestamp(last_fit)).days >= refit_days:
            qe.fit_pools(before=day, rows=rows)
            last_fit = day
        fin = qe.combine(r)
        ys.append(r["res"])
        for n in M1X2:
            evals[n].append(r[n])
        evals["ens"].append(fin["1x2"])
        if strategy == "ensemble":
            p1, pou = fin["1x2"], fin["ou25"]
        elif strategy == "market":
            p1, pou = r["mkt"], r.get("mkt_ou")
        else:
            p1, pou = r[strategy], r.get(f"{strategy}_ou")
        cands = []
        for mk, p, pm, book, close, won_fn in (
                ("1x2", p1, r["mkt"], r["o1x2"], r.get("close"), lambda i: r["res"] == i),
                ("ou25", pou, r.get("mkt_ou"), r["oou"], r.get("close_ou"), lambda i: (r["over"] == 1) == (i == 0))):
            if not p or not pm:
                continue
            bk = {k: v for k, v in book.items() if k not in ("Max", "Avg", "BF")}
            if not bk:
                continue
            bo, _ = odds.best_price(bk)
            for i, (pi, pmi, o) in enumerate(zip(p, pm, bo)):
                e = pi * o - 1
                if e >= min_edge and abs(pi - pmi) <= max_dis and lo <= o <= hi:
                    cands.append((e * np.sqrt(pi), o, won_fn(i), close[i] if close else None, e))
        if cands:
            _, o, won, cp, e = max(cands)
            bets.append({"odds": o, "won": won, "pnl": (o - 1) if won else -1.0,
                         "clv": odds.clv(o, cp) if cp else None, "edge": e})
    y = np.array(ys)
    clvs = [x["clv"] for x in bets if x["clv"] is not None]
    return {
        "strategy": strategy, "n_games": len(y), "n_bets": len(bets),
        "roi": round(float(np.mean([x["pnl"] for x in bets])), 4) if bets else 0,
        "hit_rate": round(float(np.mean([x["won"] for x in bets])), 3) if bets else None,
        "avg_clv": round(float(np.mean(clvs)), 4) if clvs else None,
        "scores": {n: metrics.summary(np.array(evals[n]), y) for n in evals},
        "weights": {"1x2": qe.pool_1x2.as_dict(), "ou25": qe.pool_ou.as_dict()},
        "test_start": test_start[:10],
    }
