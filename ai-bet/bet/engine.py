"""Junta as fontes num cartão por jogo e põe preço em todos os mercados.

Golos esperados finais (λ, μ) = média geométrica pesada de:
  mercado   λ, μ implícitos nas odds sem margem (1X2 + mais/menos 2.5)
  dc        Dixon-Coles da liga (só se as duas equipas forem da mesma liga)
  sot       Dixon-Coles nos remates enquadrados
  elo       ClubElo, para jogos entre ligas (o total de golos vem do mercado)
O mercado tem o maior peso: nos estudos, as odds são o melhor preditor que há.
Os modelos só desviam a estimativa quando têm razões para isso.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import clubelo, markets, names, odds as oddsmod, oddsapi, understat
from .models.dixon_coles import DixonColes
from .quant import QuantEngine

DEFAULT_BLEND = {"mercado": 0.70, "xg": 0.15, "dc": 0.05, "sot": 0.05, "elo": 0.05}
EXCLUDE_BEST = {"Max", "Avg", "BF", "Matchbook"}  # agregados e bolsas (com comissão) não contam como "melhor odd"


def _ts(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
    return pd.Timestamp(x).timestamp()


def build_matches(cfg, fd_fixtures: pd.DataFrame, api_events: list[dict], now: pd.Timestamp, days_ahead: float):
    """Une os jogos da Odds API com os do football-data, sem duplicados."""
    out = []
    lo, hi = now + pd.Timedelta(minutes=30), now + pd.Timedelta(days=days_ahead)
    used = set()
    for ev in api_events:
        t = pd.Timestamp(datetime.fromtimestamp(_ts(ev["commence_time"]), tz=timezone.utc)).tz_convert("Europe/Lisbon").tz_localize(None)
        if not (lo <= t <= hi):
            continue
        name, div, flag = oddsapi.comp_info(ev["sport_key"])
        bo = oddsapi.parse_event(ev)
        m = {"mid": f"{ev['sport_key']}|{t:%Y-%m-%d}|{names.key(ev['home_team'])}|{names.key(ev['away_team'])}",
             "event_id": ev["id"], "sport": ev["sport_key"], "comp": name, "flag": flag, "div": div,
             "date": t.isoformat(), "home": ev["home_team"], "away": ev["away_team"], "odds": bo,
             "source": "Odds API"}
        if div is not None and not fd_fixtures.empty:
            cand = fd_fixtures[(fd_fixtures["div"] == div)]
            for i, r in cand.iterrows():
                if abs((r["date"] - t).total_seconds()) < 30 * 3600 and names.same(r["home"], ev["home_team"]) \
                        and names.same(r["away"], ev["away_team"]):
                    _merge_fd(m["odds"], r)
                    m["fd_home"], m["fd_away"] = r["home"], r["away"]
                    used.add(i)
                    break
        out.append(m)
    if not fd_fixtures.empty:
        for i, r in fd_fixtures.iterrows():
            if i in used or not (lo <= r["date"] <= hi):
                continue
            key = oddsapi.DIV_TO_KEY.get(r["div"])
            if key is None or key not in (cfg.get("odds_api", {}).get("sports") or oddsapi.COMPETITIONS):
                continue
            name, div, flag = oddsapi.comp_info(key)
            bo = {"1x2": {}, "ou": {}, "btts": {}, "ah": {}}
            _merge_fd(bo, r)
            out.append({"mid": f"{key}|{r['date']:%Y-%m-%d}|{names.key(r['home'])}|{names.key(r['away'])}",
                        "event_id": None, "sport": key, "comp": name, "flag": flag, "div": div,
                        "date": r["date"].isoformat(), "home": r["home"], "away": r["away"],
                        "fd_home": r["home"], "fd_away": r["away"], "odds": bo, "source": "football-data"})
    out.sort(key=lambda m: m["date"])
    return out


def _merge_fd(bo, r):
    for b, o in (r.get("o1x2") or {}).items():
        bo["1x2"].setdefault(b, o)
    for b, o in (r.get("oou") or {}).items():
        bo["ou"].setdefault(2.5, {}).setdefault(b, o)


def merge_extra(m, extra_event):
    if not extra_event:
        return
    ex = oddsapi.parse_event({"home_team": m["home"], "away_team": m["away"],
                              "bookmakers": extra_event.get("bookmakers", [])})
    for k in ("btts",):
        m["odds"][k].update(ex[k])
    for k in ("ou", "ah"):
        for ln, d in ex[k].items():
            m["odds"][k].setdefault(ln, {}).update(d)


class Models:
    """Modelos de liga (Dixon-Coles e remates) + ratings Elo."""

    def __init__(self, cfg, hist: pd.DataFrame, elo: dict, now: pd.Timestamp, xg_offline=None, log=print):
        self.cfg, self.elo = cfg, elo
        self.qe = QuantEngine(cfg)
        self.by_div = {}
        self.xg = {}
        self.tables = {}
        self.hist = hist
        for div, d in hist[hist["played"]].groupby("div"):
            dc, sot = self.qe._fit_models(d, now.normalize())
            if dc is not None:
                self.by_div[div] = (dc, sot, d)
            self.tables[div] = standings(d)
        # xG real (Understat) para as 5 grandes ligas
        for div in cfg.get("leagues", []):
            if div not in understat.SLUG and xg_offline is None:
                continue
            x = understat.load(div, today=now.date(), offline=xg_offline)
            if x.empty:
                continue
            x = x[(x["date"] < now) & (x["date"] >= now - pd.Timedelta(days=self.qe.window_days))]
            if len(x) < 60:
                continue
            m = DixonColes(xi=self.qe.xi, use_rho=False).fit(x["home"], x["away"], x["xg_h"], x["xg_a"],
                                                             (now - x["date"]).dt.days.values)
            self.xg[div] = (m, x)
        if self.xg:
            log(f"  xG do Understat: {', '.join(self.xg)}")
        self.blend = {**DEFAULT_BLEND, **cfg.get("blend", {})}

    def league(self, m):
        if m.get("div") not in self.by_div:
            return None
        dc, sot, d = self.by_div[m["div"]]
        h = names.match(m.get("fd_home") or m["home"], dc.teams)
        a = names.match(m.get("fd_away") or m["away"], dc.teams)
        if not h or not a:
            return None
        return dc, sot, d, h, a

    def card(self, m: dict, adjust: dict | None = None) -> dict:
        """Calcula λ, μ de cada fonte, o λ, μ final e o preço de todos os mercados."""
        bo = m["odds"]
        comp = {}
        p1 = oddsmod.consensus({b: o for b, o in bo["1x2"].items() if b != "Max"}, "shin", sharp="PS")
        pou = None
        if 2.5 in bo["ou"]:
            pou = oddsmod.consensus({b: o for b, o in bo["ou"][2.5].items() if b != "Max"}, "shin", sharp="PS")
        rho = markets.RHO
        if p1 is not None:
            comp["mercado"] = markets.implied_rates(p1, None if pou is None else float(pou[0]))
        lg = self.league(m)
        info = {}
        if lg:
            dc, sot, d, h, a = lg
            comp["dc"] = dc.rates(h, a)
            comp["sot"] = sot.rates(h, a)
            rho = dc.rho
            info["form"] = {"home": _form(d, h), "away": _form(d, a)}
            info["rest_days"] = {"home": _rest(d, h, m["date"]), "away": _rest(d, a, m["date"])}
            info["h2h"] = _h2h(d, h, a)
            tb = self.tables.get(m["div"]) or {}
            info["table"] = {"home": tb.get(h), "away": tb.get(a), "n": len(tb)}
        if m.get("div") in self.xg:
            xm, xd = self.xg[m["div"]]
            hx = names.match(m.get("fd_home") or m["home"], xm.teams)
            ax = names.match(m.get("fd_away") or m["away"], xm.teams)
            if hx and ax:
                comp["xg"] = xm.rates(hx, ax)
                info["xg_form"] = {"home": _xg_form(xd, hx), "away": _xg_form(xd, ax)}
        eh, ea = clubelo.lookup(self.elo, m["home"]), clubelo.lookup(self.elo, m["away"])
        if eh and ea:
            info["elo"] = [round(eh), round(ea)]
            we = clubelo.expectancy(eh, ea)
            tot = sum(comp["mercado"]) if "mercado" in comp else (sum(comp["dc"]) if "dc" in comp else 2.6)
            # numa liga com DC, o Elo acrescenta pouco; brilha nos jogos entre ligas
            if "dc" not in comp:
                comp["elo"] = markets.rates_from_expectancy(we, tot, rho)
        if not comp:
            return {"ok": False}
        ws = {k: self.blend.get(k, 0) for k in comp}
        if sum(ws.values()) <= 0:
            ws = {k: 1 for k in comp}
        tw = sum(ws.values())
        lam = math.exp(sum(ws[k] * math.log(comp[k][0]) for k in comp) / tw)
        mu = math.exp(sum(ws[k] * math.log(comp[k][1]) for k in comp) / tw)
        base = (lam, mu)
        if adjust:
            lam *= adjust["lam_eff"]
            mu *= adjust["mu_eff"]
        mat = markets.matrix(lam, mu, rho)
        sels = markets.selections(mat, m["home"], m["away"])
        # lado do mercado: mesma conta com o λ, μ do mercado
        if "mercado" in comp:
            msel = {(s["market"], s["code"]): s for s in markets.selections(markets.matrix(*comp["mercado"], rho), m["home"], m["away"])}
        else:
            msel = {}
        # modelos sem mercado (para ver se "apoiam" o desvio)
        models_only = {k: v for k, v in comp.items() if k != "mercado"}
        if models_only:
            mw = {k: ws[k] for k in models_only}
            t2 = sum(mw.values()) or 1
            ml = math.exp(sum(mw[k] * math.log(v[0]) for k, v in models_only.items()) / t2)
            mm = math.exp(sum(mw[k] * math.log(v[1]) for k, v in models_only.items()) / t2)
            modsel = {(s["market"], s["code"]): s for s in markets.selections(markets.matrix(ml, mm, rho), m["home"], m["away"])}
        else:
            modsel = {}
        for s in sels:
            k = (s["market"], s["code"])
            s["p"] = markets.eff_prob(s)
            s["fair"] = markets.fair_odds(s)
            s["p_mkt"] = markets.eff_prob(msel[k]) if k in msel else None
            s["p_models"] = markets.eff_prob(modsel[k]) if k in modsel else None
            if s["market"] == "1x2" and p1 is not None:
                s["p_mkt"] = float(p1["HDA".index(s["code"])])
            if s["market"] == "dc" and p1 is not None:
                ph, pd_, pa = p1
                s["p_mkt"] = float({"1X": ph + pd_, "X2": pa + pd_, "12": ph + pa}[s["code"]])
            real = {b: o for b, o in markets.market_odds(s, bo).items() if b not in EXCLUDE_BEST and o}
            if real:
                b, o = max(real.items(), key=lambda kv: kv[1])
                s["best"], s["book"], s["n_books"] = float(o), b, len(real)
            else:
                s["best"], s["book"], s["n_books"] = None, None, 0
        flat = [(float(mat[i, j]), i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
        flat.sort(reverse=True)
        d = {(s["market"], s["code"]): s["p"] for s in sels}
        p1x2 = [d[("1x2", "H")], d[("1x2", "D")], d[("1x2", "A")]]
        k = max(range(3), key=lambda i: p1x2[i])
        verdict = [f"{m['home']} vence", "Empate", f"{m['away']} vence"][k]
        out = {"ok": True, "lam": lam, "mu": mu, "base": base, "rho": rho,
               "predict": {"p1x2": p1x2, "verdict": verdict, "verdict_p": p1x2[k],
                           "scores": [[i, j, p] for p, i, j in flat[:3]],
                           "over25": d[("ou2.5", "O")], "btts": d[("btts", "Y")]},
               "components": {k: [round(v[0], 3), round(v[1], 3)] for k, v in comp.items()},
               "weights": {k: round(ws[k] / tw, 3) for k in comp}, "sels": sels,
               "matrix": mat[:6, :6].round(4).tolist(),
               "p1x2_mkt": None if p1 is None else [float(x) for x in p1]}
        out.update(info)
        return out


def standings(d: pd.DataFrame) -> dict:
    """Classificação da época mais recente: equipa -> {pos, pts, j, dg}."""
    if d.empty or "season" not in d:
        return {}
    cur = d[d["season"] == d["season"].max()]
    t = {}
    for _, r in cur.iterrows():
        for team, gf, ga in ((r["home"], r["hg"], r["ag"]), (r["away"], r["ag"], r["hg"])):
            x = t.setdefault(team, {"pts": 0, "j": 0, "gf": 0, "ga": 0})
            x["j"] += 1; x["gf"] += gf; x["ga"] += ga
            x["pts"] += 3 if gf > ga else 1 if gf == ga else 0
    order = sorted(t, key=lambda k: (-t[k]["pts"], -(t[k]["gf"] - t[k]["ga"]), -t[k]["gf"]))
    return {k: {"pos": i + 1, "pts": int(t[k]["pts"]), "j": int(t[k]["j"]), "dg": int(t[k]["gf"] - t[k]["ga"])}
            for i, k in enumerate(order)}


def _h2h(d: pd.DataFrame, h: str, a: str, n: int = 5) -> list:
    g = d[((d["home"] == h) & (d["away"] == a)) | ((d["home"] == a) & (d["away"] == h))].tail(n)
    return [{"date": r["date"].strftime("%Y-%m-%d"), "home": r["home"], "away": r["away"],
             "score": f"{int(r['hg'])}-{int(r['ag'])}"} for _, r in g.iterrows()][::-1]


def _xg_form(x: pd.DataFrame, team: str, n: int = 6) -> dict:
    g = x[(x["home"] == team) | (x["away"] == team)].tail(n)
    xf = sum(r["xg_h"] if r["home"] == team else r["xg_a"] for _, r in g.iterrows())
    xa = sum(r["xg_a"] if r["home"] == team else r["xg_h"] for _, r in g.iterrows())
    k = max(len(g), 1)
    return {"xg_for": round(xf / k, 2), "xg_against": round(xa / k, 2), "n": len(g)}


def _form(d: pd.DataFrame, team: str, n: int = 6) -> dict:
    g = d[(d["home"] == team) | (d["away"] == team)].tail(n)
    seq, gf, ga = [], 0, 0
    for _, r in g.iterrows():
        home = r["home"] == team
        f, a = (r["hg"], r["ag"]) if home else (r["ag"], r["hg"])
        gf += f; ga += a
        seq.append("V" if f > a else ("E" if f == a else "D"))
    k = max(len(g), 1)
    return {"seq": "".join(seq), "gf": round(gf / k, 2), "ga": round(ga / k, 2)}


def _rest(d, team, when):
    g = d[((d["home"] == team) | (d["away"] == team)) & (d["date"] < pd.Timestamp(when))]
    return None if g.empty else int((pd.Timestamp(when) - g["date"].max()).days)
