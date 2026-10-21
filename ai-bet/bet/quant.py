"""Motor quantitativo. Corre sem LLM e é o chão do sistema.

Modelos:
  dc    Dixon-Coles nos golos, com decaimento temporal
  sot   Dixon-Coles em "golos sintéticos" (golos + remates enquadrados, um proxy de xG)
  pi    Pi-ratings + logit ordenado (só 1X2)
  mkt   consenso do mercado sem margem (Shin), com peso extra na Pinnacle

Tudo é avaliado em walk-forward: cada previsão usa só dados anteriores ao
jogo. Com essas previsões fora-da-amostra aprende-se a combinação (LogPool)
e medem-se as notas de cada modelo.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import db, odds
from .models.dixon_coles import DixonColes
from .models.ensemble import LogPool
from .models.pi_ratings import PiRatings

M1X2 = ["dc", "sot", "pi", "mkt"]
MOU = ["dc", "sot", "mkt"]


def _sot_goals(d: pd.DataFrame, alpha: float):
    """Golos sintéticos = alpha*golos + (1-alpha)*conversão*remates_enquadrados."""
    hst, ast = d["hst"].astype(float), d["ast"].astype(float)
    ok = hst.notna() & ast.notna()
    tot_sot = (hst[ok] + ast[ok]).sum()
    conv = (d.loc[ok, "hg"] + d.loc[ok, "ag"]).sum() / tot_sot if tot_sot > 0 else 0.3
    gh = np.where(ok, alpha * d["hg"] + (1 - alpha) * conv * hst.fillna(0), d["hg"])
    ga = np.where(ok, alpha * d["ag"] + (1 - alpha) * conv * ast.fillna(0), d["ag"])
    return gh, ga


class QuantEngine:
    def __init__(self, cfg: dict):
        q = cfg.get("quant", {})
        self.xi = q.get("xi", 0.0019)
        self.window_days = q.get("window_days", 1100)
        self.sot_alpha = q.get("sot_alpha", 0.4)
        self.devig = q.get("devig", "shin")
        self.burn_in_days = q.get("burn_in_days", 150)
        self.sharp = q.get("sharp_book", "PS")
        self.pool_1x2 = LogPool(M1X2, {"dc": 0.15, "sot": 0.15, "pi": 0.1, "mkt": 0.7})
        self.pool_ou = LogPool(MOU, {"dc": 0.15, "sot": 0.15, "mkt": 0.75})

    # ------------------------------------------------------------ helpers
    def market(self, book_odds: dict) -> np.ndarray | None:
        books = {b: o for b, o in book_odds.items() if b not in ("Max",)}
        return odds.consensus(books, self.devig, sharp=self.sharp)

    def closing_fair(self, row, market="1x2"):
        book = row["c1x2"] if market == "1x2" else row["cou"]
        if not isinstance(book, dict):
            return None
        for b in (self.sharp, "BF", "Avg"):
            if b in book and odds.valid(book[b]):
                return odds.devig(book[b], self.devig)
        return None

    def _fit_models(self, hist: pd.DataFrame, now: pd.Timestamp):
        hist = hist[(hist["date"] < now) & hist["played"]]
        hist = hist[hist["date"] >= now - pd.Timedelta(days=self.window_days)]
        if len(hist) < 60:
            return None, None
        days = (now - hist["date"]).dt.days.values
        dc = DixonColes(xi=self.xi).fit(hist["home"], hist["away"], hist["hg"], hist["ag"], days)
        gh, ga = _sot_goals(hist, self.sot_alpha)
        sot = DixonColes(xi=self.xi, use_rho=False).fit(hist["home"], hist["away"], gh, ga, days)
        sot.rho = dc.rho
        return dc, sot

    # -------------------------------------------------------- walk-forward
    def walk_forward(self, df: pd.DataFrame, log=print) -> int:
        """Gera (e guarda em cache) previsões fora-da-amostra para todos os jogos jogados."""
        done = db.oos_ids()
        new = 0
        for lg, d in df[df["played"]].groupby("div"):
            d = d.sort_values("date").reset_index(drop=True)
            start = d["date"].min() + pd.Timedelta(days=self.burn_in_days)
            todo = d[(d["date"] >= start) & (~d["mid"].isin(done))]
            if todo.empty:
                continue
            log(f"  walk-forward {lg}: {len(todo)} jogos novos")
            pi = PiRatings()
            gd_hist, res_hist = [], []
            weeks = todo["date"].dt.to_period("W").unique()
            week_models = {}
            for wk in weeks:
                t0 = wk.start_time
                week_models[wk] = self._fit_models(d, t0)
            rows = []
            todo_ids = set(todo["mid"])
            last_map = 0
            for _, r in d.iterrows():
                wk = r["date"].to_period("W")
                gd = pi.expected_gd(r["home"], r["away"])
                if r["mid"] in todo_ids and wk in week_models and week_models[wk][0] is not None:
                    if len(gd_hist) >= 200 and len(gd_hist) - last_map >= 100:
                        pi.fit_mapping(gd_hist[-1500:], res_hist[-1500:])
                        last_map = len(gd_hist)
                    dc, sot = week_models[wk]
                    rows.append(self._row_predictions(r, dc, sot, pi))
                pi.update(r["home"], r["away"], int(r["hg"]), int(r["ag"]))
                gd_hist.append(gd); res_hist.append(int(r["res"]))
            db.oos_put(rows)
            new += len(rows)
        return new

    def _row_predictions(self, r, dc, sot, pi) -> dict:
        p_dc = dc.predict(r["home"], r["away"])
        p_sot = sot.predict(r["home"], r["away"])
        out = {
            "mid": r["mid"], "div": r["div"], "date": r["date"].isoformat(),
            "home": r["home"], "away": r["away"],
            "dc": p_dc["1x2"], "sot": p_sot["1x2"], "pi": pi.predict_1x2(r["home"], r["away"]),
            "mkt": _l(self.market(r["o1x2"])),
            "dc_ou": p_dc["ou25"], "sot_ou": p_sot["ou25"], "mkt_ou": _l(self.market(r["oou"])),
            "close": _l(self.closing_fair(r, "1x2")), "close_ou": _l(self.closing_fair(r, "ou")),
            "o1x2": r["o1x2"], "oou": r["oou"],
        }
        if r.get("played"):
            out["res"] = int(r["res"])
            out["over"] = int(r["hg"] + r["ag"] > 2.5)
            out["score"] = [int(r["hg"]), int(r["ag"])]
        return out

    # -------------------------------------------------------------- pools
    def fit_pools(self, before: str | None = None, rows: list | None = None) -> dict:
        rows = rows if rows is not None else db.oos_all()
        if before:
            rows = [r for r in rows if r["date"] < before]
        r1 = [r for r in rows if r.get("mkt") and "res" in r]
        if r1:
            preds = {n: np.array([r[n] for r in r1]) for n in M1X2}
            self.pool_1x2.fit(preds, np.array([r["res"] for r in r1]))
        r2 = [r for r in rows if r.get("mkt_ou") and "over" in r]
        if r2:
            preds = {"dc": np.array([r["dc_ou"] for r in r2]), "sot": np.array([r["sot_ou"] for r in r2]),
                     "mkt": np.array([r["mkt_ou"] for r in r2])}
            self.pool_ou.fit(preds, np.array([0 if r["over"] else 1 for r in r2]))
        return {"1x2": self.pool_1x2.as_dict(), "ou25": self.pool_ou.as_dict(),
                "n": len(r1)}

    def combine(self, r: dict) -> dict:
        p1 = self.pool_1x2.predict({n: r.get(n) for n in M1X2})
        pou = self.pool_ou.predict({"dc": r.get("dc_ou"), "sot": r.get("sot_ou"), "mkt": r.get("mkt_ou")})
        return {"1x2": p1.tolist(), "ou25": pou.tolist()}

    # ------------------------------------------------------------ live
    def predict_fixtures(self, hist: pd.DataFrame, fixtures: pd.DataFrame) -> list[dict]:
        out = []
        for lg, fx in fixtures.groupby("div"):
            d = hist[(hist["div"] == lg) & hist["played"]].sort_values("date")
            if d.empty:
                continue
            now = pd.Timestamp(fx["date"].min()).normalize()
            dc, sot = self._fit_models(d, now)
            if dc is None:
                continue
            pi = PiRatings()
            gds, res = [], []
            for _, r in d.iterrows():
                gds.append(pi.expected_gd(r["home"], r["away"])); res.append(int(r["res"]))
                pi.update(r["home"], r["away"], int(r["hg"]), int(r["ag"]))
            pi.fit_mapping(gds[-1500:], res[-1500:])
            for _, r in fx.iterrows():
                row = self._row_predictions(r, dc, sot, pi)
                full = dc.predict(r["home"], r["away"])
                row["xg"] = full["xg"]
                row["rho"] = dc.rho
                row["rest_days"] = {"home": _rest(d, r["home"], r["date"]), "away": _rest(d, r["away"], r["date"])}
                row["top_scores"] = full["top_scores"]
                row["matrix"] = full["matrix"]
                row["btts"] = full["btts"]
                row["unknown_team"] = not (dc.knows(r["home"]) and dc.knows(r["away"]))
                row["form"] = {"home": _form(d, r["home"]), "away": _form(d, r["away"])}
                row["pi_ratings"] = {"home": pi.rating(r["home"]), "away": pi.rating(r["away"])}
                row["referee"] = r.get("referee")
                row["final"] = self.combine(row)
                out.append(row)
        return out


def _rest(d: pd.DataFrame, team: str, when) -> int | None:
    g = d[((d["home"] == team) | (d["away"] == team)) & (d["date"] < when)]
    if g.empty:
        return None
    return int((pd.Timestamp(when) - g["date"].max()).days)


def _l(x):
    return None if x is None else [float(v) for v in x]


def _form(d: pd.DataFrame, team: str, n: int = 6) -> dict:
    g = d[(d["home"] == team) | (d["away"] == team)].tail(n)
    seq, gf, ga, sf, sa = [], 0, 0, 0, 0
    for _, r in g.iterrows():
        home = r["home"] == team
        f, a = (r["hg"], r["ag"]) if home else (r["ag"], r["hg"])
        s1, s2 = (r["hst"], r["ast"]) if home else (r["ast"], r["hst"])
        gf += f; ga += a
        sf += s1 if s1 == s1 and s1 is not None else 0
        sa += s2 if s2 == s2 and s2 is not None else 0
        seq.append("V" if f > a else ("E" if f == a else "D"))
    k = max(len(g), 1)
    return {"seq": "".join(seq), "gf": round(gf / k, 2), "ga": round(ga / k, 2),
            "sot_f": round(sf / k, 2), "sot_a": round(sa / k, 2)}


def dumps(x) -> str:
    return json.dumps(x, default=lambda o: o.isoformat() if isinstance(o, (datetime, pd.Timestamp)) else str(o))
