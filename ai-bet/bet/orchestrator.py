"""O ciclo: dados → modelos → triagem → batedores → analista → advogado do diabo →
presidente → boletim do dia. Emite eventos para os "bastidores" do dashboard."""
from __future__ import annotations

import concurrent.futures as cf
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import agents, boletim, clubelo, data, db, engine, names, oddsapi, odds as oddsmod


class Run:
    def __init__(self, cfg: dict, now: datetime | None = None, world=None, log=print):
        self.cfg, self.world, self.log = cfg, world, log
        self.now = pd.Timestamp(now or datetime.now())
        self.id = self.now.strftime("%Y%m%d-%H%M") + "-" + uuid.uuid4().hex[:4]
        self.offline = Path(world.api_dir) if world else None
        self.elo_offline = Path(world.raw) if world else None

    def ev(self, agent, kind, text, mid=None, data_=None):
        db.event(self.id, agent, kind, text, mid, data_)
        if kind in ("done", "phase", "error", "pick"):
            self.log(f"  [{agent}] {text}")

    # --------------------------------------------------------------- main
    def go(self, refresh: bool = True):
        cfg = self.cfg
        oddsapi.Quota.spent_run = 0
        db.kv_set("current_run", {"id": self.id, "started": time.time(), "now": self.now.isoformat(), "status": "a correr"})
        self.ev("sistema", "phase", "Recolha", data_={"phase": 1})

        # ---------- dados
        self.ev("arquivista", "working", "Resultados, remates e odds das ligas (football-data.co.uk)")
        hist = data.load_history(cfg["leagues"], cfg.get("seasons", 4), refresh=refresh, today=self.now.date())
        fx = data.load_fixtures(cfg["leagues"], cfg.get("days_ahead", 3), refresh=refresh, now=self.now)
        self.ev("arquivista", "done", f"{int(hist['played'].sum()) if not hist.empty else 0} jogos de histórico")
        self.settle(hist)

        self.ev("corretor", "working", "A recolher odds (Odds API)")
        evs = oddsapi.fetch_window(cfg, cfg.get("days_ahead", 3) * 24, self.now.timestamp() if self.world else None,
                                   log=self.log, offline_dir=self.offline)
        self.ev("corretor", "done", f"{len(evs)} jogos com odds" + (f" · créditos restantes {oddsapi.Quota.remaining}"
                                                                    if oddsapi.Quota.remaining is not None else ""))
        self.ev("elo", "working", "Ratings ClubElo")
        elo = clubelo.ratings(self.now.date(), offline=self.elo_offline)
        self.ev("elo", "done", f"{len(elo)} clubes com rating")

        matches = engine.build_matches(cfg, fx, evs, self.now, cfg.get("days_ahead", 3))
        if not matches:
            self.ev("sistema", "done", "Não há jogos das competições escolhidas nos próximos dias.")
            bol = {"run_id": self.id, "ts": time.time(), "now": self.now.isoformat(), "singles": [], "multi": None,
                   "intro": "Não há jogos das competições escolhidas nos próximos dias.", "n_matches": 0, "n_deep": 0}
            db.kv_set("boletim", bol)
            db.kv_set("matches", [])
            return self._finish("sem jogos", bol)

        # ---------- modelos
        self.ev("sistema", "phase", "Modelos", data_={"phase": 2})
        self.ev("modelos", "working", "Mercado, xG (Understat), Dixon-Coles, remates e Elo")
        models = engine.Models(cfg, hist if not hist.empty else _empty_hist(), elo, self.now,
                               xg_offline=Path(self.world.raw) if self.world else None, log=self.log)
        for m in matches:
            m["card"] = models.card(m)
            if not m["card"]["ok"]:
                m["cands"], m["triage"] = [], -1
                continue
            m["cands"] = boletim.candidates(m, m["card"], cfg)
            ok = [c for c in m["cands"] if "reject" not in c]
            m["triage"] = ok[0]["score"] if ok else -1
            db.layer_put(m["mid"], "mercado", "1x2", m["card"].get("p1x2_mkt") or _p1x2(m["card"]))
            db.layer_put(m["mid"], "modelos", "1x2", _p1x2(m["card"]))
            db.layer_put(m["mid"], "agentes", "1x2", _p1x2(m["card"]))
        n_val = sum(1 for m in matches if m["triage"] > 0)
        self.ev("modelos", "done", f"{len(matches)} jogos com preço em todos os mercados · {n_val} com valor à partida")

        # ---------- radar
        self.ev("sistema", "phase", "Notícias", data_={"phase": 3})
        flagged = set()
        by_comp = {}
        for m in matches:
            by_comp.setdefault(m["comp"], []).append(m)
        for comp, grp in by_comp.items():
            self.ev("radar", "working", f"Notícias: {comp} ({len(grp)} jogos)")
            try:
                res = agents.radar(comp, grp, cfg, self.world) or {}
                for fl in res.get("flags", []):
                    i = fl.get("i")
                    if isinstance(i, int) and 0 <= i < len(grp) and int(fl.get("impact", 0)) >= 2:
                        grp[i]["radar"] = fl
                        flagged.add(grp[i]["mid"])
            except Exception as e:  # noqa: BLE001
                self.ev("radar", "error", f"{comp}: {str(e)[:140]}")
        self.ev("radar", "done", f"{len(flagged)} jogo(s) com notícia forte")

        # ---------- análise profunda
        llm_cfg = cfg["llm"]
        bc = cfg.get("boletim", {})
        pick_until = self.now + pd.Timedelta(hours=bc.get("pick_window_hours", 72))
        deep_until = self.now + pd.Timedelta(hours=llm_cfg.get("deep_window_hours", 72))
        in_window = [m for m in matches if m["card"]["ok"] and pd.Timestamp(m["date"]) <= pick_until]
        elig = [m for m in in_window if pd.Timestamp(m["date"]) <= deep_until]
        # prioridade: notícias fortes, depois jogos com apostas candidatas, depois os mais próximos
        elig.sort(key=lambda m: (m["mid"] not in flagged, -m["triage"], m["date"]))
        deep = elig[: llm_cfg.get("max_deep_matches", 8)]
        deep_ids = {m["mid"] for m in deep}
        for m in matches:
            m["stage"] = "analise" if m["mid"] in deep_ids else "triagem"
        self.ev("triagem", "done", f"{len(deep)} jogos para análise a fundo", data_={"deep": list(deep_ids)})
        with cf.ThreadPoolExecutor(max_workers=max(1, llm_cfg.get("parallel", 3))) as ex:
            list(ex.map(lambda m: self._deep(m, models), deep))

        # ---------- advogado do diabo
        self.ev("sistema", "phase", "Tribunal", data_={"phase": 4})
        pool = []
        for m in in_window:
            ok = [c for c in m["cands"] if "reject" not in c][:3]
            if not ok:
                continue
            if m["mid"] in deep_ids:
                self.ev("cetico", "working", f"A atacar {len(ok)} aposta(s) em {m['home']}–{m['away']}", m["mid"])
                try:
                    rev = agents.cetico(m, ok, cfg) or {"reviews": []}
                except Exception as e:  # noqa: BLE001
                    rev = {"reviews": []}
                    self.ev("cetico", "error", f"falhou: {e}", m["mid"])
                for r in rev.get("reviews", []):
                    i = r.get("id")
                    if isinstance(i, int) and 0 <= i < len(ok):
                        ok[i]["skeptic"] = {"severity": int(r.get("severity", 0)), "argument": r.get("argument", "")}
                        if ok[i]["skeptic"]["severity"] >= 3:
                            ok[i]["reject"] = "chumbada pelo advogado do diabo"
                self.ev("cetico", "done", f"{m['home']}–{m['away']}: " + ((ok[0].get("skeptic") or {}).get("argument") or "revisto"), m["mid"])
            for c in ok:
                if "reject" in c:
                    continue
                c["stars"] = boletim.stars(c, m.get("has_news", False))
                if m["mid"] not in deep_ids:  # sem verificação de notícias: confiança mínima
                    c["stars"] = 1
                    c["unchecked"] = True
                self._decorate(c, m)
                pool.append(c)
        # as mais prováveis (odds baixas, muita probabilidade)
        safe = []
        for m, c in boletim.safe_picks(in_window, cfg):
            c["stars"] = 3 if c["p"] >= 0.82 else 2
            if m["mid"] not in deep_ids:
                c["unchecked"] = True
            self._decorate(c, m)
            safe.append(c)

        # ---------- proposta + presidente
        self.ev("sistema", "phase", "Boletim", data_={"phase": 5})
        singles = boletim.pick_slate(pool, cfg)
        allc = pool + [c for c in safe if c["key"] not in {x["key"] for x in pool}]
        multi = boletim.build_multi(allc, cfg)
        multi_ids = [next(i for i, c in enumerate(allc) if c["key"] == k) for k in multi["legs"]] if multi else []
        safe_ids = [next(i for i, c in enumerate(allc) if c["key"] == x["key"]) for x in safe]
        self.ev("presidente", "working", f"A fechar o boletim ({len(pool)} candidatas, {len(safe)} favoritas)")
        try:
            dec = agents.presidente(allc, {"singles": singles, "multi": multi_ids, "safe": safe_ids}, cfg) or {}
        except Exception as e:  # noqa: BLE001
            self.ev("presidente", "error", f"indisponível: {e}")
            dec = {}
        s_ids, sm = [], set()
        for i in dec.get("singles") or singles:
            if isinstance(i, int) and 0 <= i < len(pool) and pool[i]["mid"] not in sm:
                s_ids.append(i); sm.add(pool[i]["mid"])
        for i in singles:  # completa com a proposta do sistema se o presidente escolheu poucas
            if len(s_ids) >= cfg.get("boletim", {}).get("singles", 5):
                break
            if i not in s_ids and pool[i]["mid"] not in sm:
                s_ids.append(i); sm.add(pool[i]["mid"])
        m_ids = [i for i in (dec.get("multi") or multi_ids) if isinstance(i, int) and 0 <= i < len(allc)]
        if len(m_ids) < 2 or len({allc[i]["mid"] for i in m_ids}) < len(m_ids):
            m_ids = multi_ids
        why = {str(k): v for k, v in (dec.get("why") or {}).items()}

        out_singles, saved = [], set()
        for i in s_ids:
            c = allc[i]
            c["why"] = why.get(str(i)) or ""
            out_singles.append(_public_pick(c))
            db.pick_put(_pick_row(self.id, c, "single")); saved.add(c["key"])
            self.ev("presidente", "pick", f"{c['label']} · {c['home']}–{c['away']} @ {c['odds']:.2f} {'★' * c['stars']}", c["mid"])
        out_safe = []
        for i in safe_ids:
            c = allc[i]
            c["why"] = why.get(str(i)) or c.get("why") or ""
            out_safe.append(_public_pick(c))
            if c["key"] not in saved:
                db.pick_put(_pick_row(self.id, c, "safe")); saved.add(c["key"])
        out_multi = None
        if len(m_ids) >= 2:
            legs = [allc[i] for i in m_ids]
            for i in m_ids:
                allc[i].setdefault("why", why.get(str(i)) or "")
                if allc[i]["key"] not in saved:
                    db.pick_put(_pick_row(self.id, allc[i], "leg")); saved.add(allc[i]["key"])
            prob = math.prod(c["p"] for c in legs)
            o = math.prod(c["odds"] for c in legs)
            out_multi = {"legs": [_public_pick(c) for c in legs], "odds": round(o, 2), "prob": prob,
                         "fair": round(1 / prob, 2), "min": round(math.prod(c["min"] for c in legs), 2),
                         "edge": prob * o - 1, "estimated": any(c.get("estimated") for c in legs)}
            mkey = "MULTI::" + "::".join(sorted(c["key"] for c in legs))
            out_multi["key"] = mkey
            db.pick_put({"key": mkey, "run_id": self.id, "kind": "multi", "mid": ";".join(c["mid"] for c in legs),
                         "comp": "Múltipla", "flag": "🎟️", "date": max(c["date"] for c in legs),
                         "home": " + ".join(c["label"] for c in legs), "away": "", "market": "multi", "code": "",
                         "label": f"Múltipla de {len(legs)}", "p": prob, "fair": 1 / prob, "min_odds": out_multi["min"],
                         "odds": o, "estimated": int(out_multi["estimated"]), "edge": out_multi["edge"],
                         "stars": min(c["stars"] for c in legs)})
            self.ev("presidente", "pick", f"Múltipla de {len(legs)} @ {o:.2f}")
        bol = {"run_id": self.id, "ts": time.time(), "now": self.now.isoformat(), "date": self.now.strftime("%Y-%m-%d"),
               "intro": dec.get("intro") or "", "singles": out_singles, "safe": out_safe, "multi": out_multi,
               "n_matches": len(matches), "n_deep": len(deep), "credits": oddsapi.Quota.remaining,
               "comps": sorted(by_comp), "demo": bool(self.world)}
        db.boletim_put(self.id, bol)
        db.kv_set("boletim", bol)
        db.kv_set("matches", [_public_match(m) for m in matches])
        self.ev("presidente", "done", bol["intro"] or "Boletim fechado")
        return self._finish("ok", bol)

    def _decorate(self, c, m):
        card = m.get("card") or {}
        c.update({"home": m["home"], "away": m["away"], "comp": m["comp"], "flag": m["flag"],
                  "date": m["date"], "sport": m["sport"],
                  "news_reasons": (m.get("adjust") or {}).get("reasons") if m.get("has_news") else None,
                  "form_txt": _form_txt(m), "table": card.get("table"), "xg_form": card.get("xg_form"),
                  "h2h": card.get("h2h")})

    # ------------------------------------------------------------- deep
    def _deep(self, m, models):
        mid = m["mid"]
        if m.get("event_id"):
            extra = oddsapi.fetch_extra(self.cfg, m["sport"], m["event_id"], log=self.log, offline_dir=self.offline)
            if extra:
                engine.merge_extra(m, extra)
                self.ev("corretor", "done", f"{m['home']}–{m['away']}: odds de ambas marcam e handicap", mid)
        news = self.world.news_for(m.get("div"), m["home"], m["away"]) if self.world else None
        scouts = {}
        for who, fn in (("medico", agents.medico), ("reporter", agents.reporter)):
            self.ev(who, "working", f"{m['home']}–{m['away']}", mid)
            try:
                scouts[who] = fn(m, self.cfg, news) or {}
                self.ev(who, "done", _scout_summary(who, scouts[who], m), mid)
            except Exception as e:  # noqa: BLE001
                scouts[who] = {}
                self.ev(who, "error", f"falhou: {str(e)[:150]}", mid)
        m["scouts"] = scouts
        k = self.cfg["llm"].get("adjuster_samples", 3)
        self.ev("analista", "working", f"{k} palpites para {m['home']}–{m['away']}", mid)
        samples = []
        for s in range(k):
            try:
                samples.append(agents.ajustador(m, scouts, self.cfg, seed=s))
            except Exception as e:  # noqa: BLE001
                self.ev("analista", "error", f"palpite {s + 1} falhou: {str(e)[:120]}", mid)
        adj = agents.aggregate_adjust(samples)
        nw = self.cfg.get("quant", {}).get("news_weight", 0.8)
        keep = (1 - adj["priced_in"]) * nw if adj["n"] else 0.0
        adj["lam_eff"] = adj["lam"] ** keep
        adj["mu_eff"] = adj["mu"] ** keep
        m["adjust"] = adj
        m["has_news"] = abs(adj["lam_eff"] - 1) > 0.01 or abs(adj["mu_eff"] - 1) > 0.01
        old = m["card"]
        m["card"] = models.card(m, adj)
        shift = abs(m["card"]["lam"] - old["lam"]) + abs(m["card"]["mu"] - old["mu"])
        m["cands"] = boletim.candidates(m, m["card"], self.cfg, slack=min(0.08, shift * 0.15) if m["has_news"] else 0.0)
        db.layer_put(mid, "agentes", "1x2", _p1x2(m["card"]))
        ok = [c for c in m["cands"] if "reject" not in c]
        txt = (f"xG {old['lam']:.2f}→{m['card']['lam']:.2f} / {old['mu']:.2f}→{m['card']['mu']:.2f} · {adj['reasons'][0]}"
               if m["has_news"] and adj["reasons"] else "sem notícias que mexam nas contas")
        self.ev("analista", "done", f"{m['home']}–{m['away']}: {txt} · {len(ok)} aposta(s) com valor", mid)

    # ----------------------------------------------------------- settle
    def settle(self, hist: pd.DataFrame) -> int:
        open_ = db.picks("open")
        if not open_:
            return 0
        played = hist[hist["played"]] if not hist.empty else hist
        need_api = sorted({p["sport"] for p in open_ if p["kind"] in ("single", "leg", "safe") and p["sport"]
                           and oddsapi.COMPETITIONS.get(p["sport"], (None, None))[1] not in self.cfg["leagues"]
                           and pd.Timestamp(p["date"]) < self.now - pd.Timedelta(hours=2)})
        scores = oddsapi.fetch_scores(self.cfg, need_api, log=self.log, offline_dir=self.offline) if need_api else []
        n = 0
        for p in [x for x in open_ if x["kind"] in ("single", "leg", "safe")]:
            if pd.Timestamp(p["date"]) > self.now - pd.Timedelta(hours=2):
                continue
            r = self._find_result(p, played, scores)
            if not r:
                continue
            hg, ag, close = r
            w, pu, l = boletim.settle_sel(p["market"], p["code"], p["line"], hg, ag)
            units = boletim.profit_units(w, pu, l, p["odds"])
            status = ("won" if w > 0.99 else "lost" if l > 0.99 else "void" if pu > 0.99
                      else "half_won" if w > l else "half_lost")
            clv = None
            if close and p["market"] in close and not p["estimated"]:
                i = "HDA".index(p["code"]) if p["market"] == "1x2" else (0 if p["code"] == "O" else 1)
                clv = oddsmod.clv(p["odds"], float(close[p["market"]][i]))
            db.pick_settle(p["key"], status, f"{hg}-{ag}", round(units, 3), clv)
            n += 1
        settled = {p["key"]: p for p in db.picks() if p["kind"] in ("single", "leg", "safe")}
        for p in [x for x in db.picks("open") if x["kind"] == "multi"]:
            legs = p["key"].split("::")[1:]
            done = [settled[k] for k in legs if k in settled and settled[k]["status"] != "open"]
            if len(done) < len(legs):
                continue
            if any(d["status"] == "lost" for d in done):
                db.pick_settle(p["key"], "lost", "", -1.0)
            else:
                mult = math.prod(1 + (d["units"] or 0) for d in done)
                db.pick_settle(p["key"], "won" if mult > 1.0001 else ("void" if abs(mult - 1) < 1e-4 else "lost"), "", round(mult - 1, 3))
            n += 1
        # resultados para o boletim de notas
        if not played.empty:
            for _, r in played[played["date"] >= self.now - pd.Timedelta(days=10)].iterrows():
                key = oddsapi.DIV_TO_KEY.get(r["div"])
                if key:
                    db.layer_set_result(f"{key}|{r['date']:%Y-%m-%d}|{names.key(r['home'])}|{names.key(r['away'])}", "1x2",
                                        int(r["res"]))
        for s in scores:
            if s.get("completed") and s.get("scores"):
                sc = {x["name"]: int(x["score"]) for x in s["scores"]}
                if s["home_team"] in sc and s["away_team"] in sc:
                    hg, ag = sc[s["home_team"]], sc[s["away_team"]]
                    t = pd.Timestamp(s["commence_time"]).tz_convert("Europe/Lisbon").tz_localize(None)
                    db.layer_set_result(f"{s['sport_key']}|{t:%Y-%m-%d}|{names.key(s['home_team'])}|{names.key(s['away_team'])}",
                                        "1x2", 0 if hg > ag else 1 if hg == ag else 2)
        if n:
            self.ev("arquivista", "done", f"{n} apostas do historial fechadas")
        return n

    def _find_result(self, p, played, scores):
        div = oddsapi.COMPETITIONS.get(p["sport"], (None, None))[1]
        if div and not played.empty:
            c = played[played["div"] == div]
            c = c[(c["date"] - pd.Timestamp(p["date"])).abs() <= pd.Timedelta(hours=30)]
            for _, r in c.iterrows():
                if names.same(r["home"], p["home"]) and names.same(r["away"], p["away"]):
                    return int(r["hg"]), int(r["ag"]), _closing(r)
        for s in scores:
            if not s.get("completed") or not s.get("scores"):
                continue
            if names.same(s["home_team"], p["home"]) and names.same(s["away_team"], p["away"]):
                sc = {x["name"]: int(x["score"]) for x in s["scores"]}
                if s["home_team"] in sc and s["away_team"] in sc:
                    return sc[s["home_team"]], sc[s["away_team"]], None
        return None

    def _finish(self, status, bol):
        db.kv_set("current_run", {"id": self.id, "finished": time.time(), "now": self.now.isoformat(),
                                  "status": status, "picks": len(bol.get("singles", []))})
        self.ev("sistema", "phase", "Concluído", data_={"phase": 6})
        return {"run_id": self.id, "status": status, "apostas": len(bol.get("singles", [])),
                "multipla": bool(bol.get("multi"))}


def _empty_hist():
    return pd.DataFrame({"played": pd.Series(dtype=bool), "div": pd.Series(dtype=str)})


def _closing(r):
    out = {}
    for mk, col in (("1x2", "c1x2"), ("ou2.5", "cou")):
        book = r.get(col)
        if isinstance(book, dict):
            for b in ("PS", "BF", "Avg"):
                if b in book and oddsmod.valid(book[b]):
                    out[mk] = oddsmod.devig(book[b], "shin")
                    break
    return out


def _p1x2(card):
    d = {(s["market"], s["code"]): s["p"] for s in card["sels"]}
    return [d[("1x2", "H")], d[("1x2", "D")], d[("1x2", "A")]]


def _form_txt(m):
    f = (m.get("card") or {}).get("form")
    if not f:
        return None
    return f"forma {m['home']} {f['home']['seq'] or '—'}, {m['away']} {f['away']['seq'] or '—'}"


def _scout_summary(who, s, m):
    if who == "medico":
        h = len((s.get("home") or {}).get("absences", []))
        a = len((s.get("away") or {}).get("absences", []))
        return f"{m['home']}: {h} baixa(s) · {m['away']}: {a} baixa(s)"
    notes = ((s.get("home") or {}).get("notes") or []) + ((s.get("away") or {}).get("notes") or [])
    return notes[0] if notes else "sem novidades de contexto"


PUB = ["key", "mid", "market", "code", "line", "label", "p", "p_mkt", "p_models", "fair", "min", "odds", "book",
       "n_books", "estimated", "edge", "stars", "why", "skeptic", "home", "away", "comp", "flag", "date", "news_reasons",
       "kind", "unchecked", "vs_mkt"]


def _public_pick(c):
    return {k: c.get(k) for k in PUB}


def _pick_row(run_id, c, kind):
    return {"key": c["key"], "run_id": run_id, "kind": kind, "mid": c["mid"], "sport": c["sport"], "comp": c["comp"],
            "flag": c["flag"], "date": c["date"], "home": c["home"], "away": c["away"], "market": c["market"],
            "code": c["code"], "line": c.get("line"), "label": c["label"], "p": c["p"], "p_mkt": c["p_mkt"],
            "fair": c["fair"], "min_odds": c["min"], "odds": c["odds"], "book": c.get("book"),
            "estimated": int(bool(c.get("estimated"))), "edge": c["edge"], "stars": c["stars"], "why": c.get("why")}


def _public_match(m):
    card = m.get("card") or {}
    cands = {c["key"]: c for c in m.get("cands", [])}
    sels = []
    for s in card.get("sels", []):
        if s["market"].startswith("ah") and not (1.5 <= s["fair"] <= 2.7):
            continue
        d = {k: s.get(k) for k in ("market", "code", "line", "label", "p", "p_mkt", "p_models", "fair", "best", "book")}
        c = cands.get(f"{m['mid']}#{s['market']}#{s['code']}")
        if c:
            d.update({"min": c.get("min"), "edge": c.get("edge"), "reject": c.get("reject"), "estimated": c.get("estimated"), "kind": c.get("kind"),
                      "skeptic": c.get("skeptic")})
        sels.append(d)
    best = next((c for c in m.get("cands", []) if "reject" not in c), None)
    return {"mid": m["mid"], "comp": m["comp"], "flag": m["flag"], "sport": m["sport"], "date": m["date"],
            "home": m["home"], "away": m["away"], "stage": m.get("stage"), "radar": m.get("radar"),
            "xg": [card.get("lam"), card.get("mu")], "components": card.get("components"),
            "weights": card.get("weights"), "form": card.get("form"), "rest_days": card.get("rest_days"),
            "elo": card.get("elo"), "matrix": card.get("matrix"), "scouts": m.get("scouts"),
            "adjust": m.get("adjust"), "has_news": m.get("has_news"), "sels": sels,
            "predict": card.get("predict"), "table": card.get("table"), "h2h": card.get("h2h"),
            "xg_form": card.get("xg_form"),
            "best": {k: best.get(k) for k in ("label", "odds", "edge", "estimated", "min")} if best else None}
