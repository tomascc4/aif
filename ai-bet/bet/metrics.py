"""Métricas de previsão: log loss, Brier, RPS (Ranked Probability Score, a
métrica clássica do 1X2, Constantinou & Fenton 2012) e curva de calibração."""
from __future__ import annotations

import numpy as np


def log_loss(p, y) -> float:
    p = np.asarray(p, float); y = np.asarray(y, int)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(p, y) -> float:
    p = np.asarray(p, float); y = np.asarray(y, int)
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return float(((p - o) ** 2).sum(1).mean())


def rps(p, y) -> float:
    p = np.asarray(p, float); y = np.asarray(y, int)
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    cp, co = np.cumsum(p, 1)[:, :-1], np.cumsum(o, 1)[:, :-1]
    return float(((cp - co) ** 2).sum(1).mean() / (p.shape[1] - 1))


def calibration(p, y, bins: int = 10):
    """Curva de calibração sobre todos os resultados (one-vs-rest)."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    pf, of = p.ravel(), o.ravel()
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pf >= a) & (pf < b)
        if m.sum() >= 10:
            out.append({"pred": float(pf[m].mean()), "obs": float(of[m].mean()), "n": int(m.sum())})
    return out


def summary(p, y) -> dict:
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "logloss": round(log_loss(p, y), 4),
            "brier": round(brier(p, y), 4), "rps": round(rps(p, y), 4)}
