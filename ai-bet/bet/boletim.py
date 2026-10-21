"""O BOLETIM DO DIA: escolher apostas concretas, montar a múltipla e, depois
dos jogos, ver o que bateu certo.

Há dois tipos de aposta no boletim:

  VALOR    há uma odd REAL, numa casa, acima da nossa odd mínima (edge >= min_edge).
           É o que, em teoria, dá lucro a longo prazo. Num mercado eficiente
           aparecem poucas por dia.
  SÓLIDA   seleção provável (>= solid_min_prob) em que o sistema (modelos + xG +
           notícias) concorda com o mercado ou está acima dele. Não há garantia
           de valor: só vale a pena se a tua casa pagar pelo menos a "odd mínima",
           que aqui é a odd justa + 1%.

O boletim junta as melhores das duas, no máximo uma por jogo e sem repetir
demasiado o mesmo tipo de mercado, para nunca vir vazio quando há jogos.
"""
from __future__ import annotations

import itertools
import math

from . import markets


def _family(market):
    return "ah" if market.startswith("ah") else "ou" if market.startswith("ou") else market


def candidates(m: dict, card: dict, cfg: dict, slack: float = 0.0) -> list[dict]:
    b = cfg.get("boletim", {})
    min_edge = b.get("min_edge", 0.03)
    max_dis = b.get("max_disagree", 0.08) + slack
    min_p = b.get("min_prob", 0.33)
    solid_p = b.get("solid_min_prob", 0.55)
    solid_tol = b.get("solid_tolerance", 0.01)
    lo, hi = b.get("min_odds", 1.25), b.get("max_odds", 4.5)
    out = []
    for s in card["sels"]:
        if s["p_mkt"] is None:
            continue
        fam = _family(s["market"])
        if fam == "ah" and not (1.5 <= s["fair"] <= 2.6):
            continue  # só linhas principais de handicap
        c = dict(s)
        c["mid"] = m["mid"]
        c["key"] = f"{m['mid']}#{s['market']}#{s['code']}"
        c["vs_mkt"] = s["p"] / max(s["p_mkt"], 1e-6) - 1  # quanto acima do mercado está a nossa probabilidade
        c["edge"] = markets.ev(s, s["best"]) if s["best"] else None
        reason = None
        if s["p"] < min_p:
            reason = "pouco provável"
        elif not (lo <= s["fair"] <= hi):
            reason = "odd fora do intervalo"
        elif abs(s["p"] - s["p_mkt"]) > max_dis:
            reason = "discorda demasiado do mercado"
        if reason is None and c["edge"] is not None and c["edge"] >= min_edge and c["edge"] <= 0.30:
            c["kind"] = "valor"
            c["min"] = markets.min_odds(s, min_edge)
            c["odds"], c["estimated"] = s["best"], False
            c["score"] = 1.0 + c["edge"] * s["p"] ** 0.8
        elif reason is None and s["p"] >= solid_p and c["vs_mkt"] >= -solid_tol and s["fair"] <= 2.2:
            c["kind"] = "solida"
            c["min"] = markets.min_odds(s, 0.01)
            c["odds"] = s["best"] if s["best"] else round(1 / max(s["p_mkt"], 1e-6) / 1.05, 2)
            c["estimated"] = not bool(s["best"])
            c["score"] = 0.5 * s["p"] + 2.0 * max(c["vs_mkt"], 0) + (0.2 if s["best"] and s["best"] >= c["min"] else 0)
        else:
            c["reject"] = reason or "sem valor nem confiança suficiente"
            c["score"] = -1
        if c.get("edge") is None:
            c["edge"] = markets.ev(s, c["odds"]) if c.get("odds") else None
        out.append(c)
    out.sort(key=lambda c: -c["score"])
    return out


def stars(c: dict, news: bool) -> int:
    st = 1
    if c["kind"] == "valor" and (c["edge"] or 0) >= 0.06:
        st += 1
    if c["kind"] == "solida" and c["p"] >= 0.68:
        st += 1
    pm = c.get("p_models")
    supp = (pm is not None and pm - c["p_mkt"] > 0.015) or news
    if supp:
        st += 1
    if (c.get("skeptic") or {}).get("severity", 0) >= 1:
        st -= 1
    if c["p"] < 0.40:
        st = min(st, 2)
    return max(1, min(3, st))


def pick_slate(pool: list[dict], cfg: dict) -> list[int]:
    """Escolhe as simples: valor primeiro, depois as sólidas; 1 por jogo, variedade de mercados."""
    b = cfg.get("boletim", {})
    n = b.get("singles", 5)
    max_fam = b.get("max_same_market", 2)
    order = sorted(range(len(pool)), key=lambda i: (pool[i]["kind"] != "valor", -pool[i]["stars"], -pool[i]["score"]))
    chosen, mids, fams = [], set(), {}
    for i in order:
        c = pool[i]
        f = _family(c["market"])
        if c["mid"] in mids or fams.get(f, 0) >= max_fam:
            continue
        chosen.append(i); mids.add(c["mid"]); fams[f] = fams.get(f, 0) + 1
        if len(chosen) >= n:
            break
    if len(chosen) < n:  # se a variedade não chegou, completa
        for i in order:
            if i not in chosen and pool[i]["mid"] not in mids:
                chosen.append(i); mids.add(pool[i]["mid"])
                if len(chosen) >= n:
                    break
    return chosen


def safe_picks(matches: list[dict], cfg: dict) -> list[dict]:
    """AS MAIS PROVÁVEIS: em cada jogo, a seleção com maior probabilidade de acertar
    (>= safe_min_prob), desde que o sistema não esteja abaixo do mercado.
    Odds baixas: acertam muitas vezes mas só dão lucro se a probabilidade real
    for maior do que a implícita na odd. Por isso a "mínima" aqui é a odd justa."""
    b = cfg.get("boletim", {})
    pmin = b.get("safe_min_prob", 0.72)
    out = []
    # resultado primeiro (é o que se costuma apostar), depois dupla hipótese, depois golos
    prefer = {"1x2": 0.10, "dc": 0.02, "ou1.5": 0.0, "ou2.5": 0.03, "btts": 0.02}
    for m in matches:
        card = m.get("card") or {}
        best = None
        for s in card.get("sels", []):
            if s["market"] not in prefer or s["p_mkt"] is None or s["p"] < pmin or s["fair"] < 1.12:
                continue
            if s["p"] / max(s["p_mkt"], 1e-6) - 1 < -0.02:
                continue
            sc = s["p"] + prefer.get(s["market"], 0)
            if best is None or sc > best[0]:
                best = (sc, s)
        if best:
            s = best[1]
            c = dict(s)
            c.update({"mid": m["mid"], "key": f"{m['mid']}#{s['market']}#{s['code']}", "kind": "segura",
                      "vs_mkt": s["p"] / max(s["p_mkt"], 1e-6) - 1, "min": s["fair"],
                      "odds": s["best"] if s["best"] else round(1 / max(s["p_mkt"], 1e-6) / 1.05, 2),
                      "estimated": not bool(s["best"])})
            c["edge"] = markets.ev(s, c["odds"])
            c["score"] = s["p"]
            out.append((m, c))
    out.sort(key=lambda mc: -(mc[1]["p"] + prefer.get(mc[1]["market"], 0)))
    res, fams = [], {}
    for m, c in out:  # variedade: no máximo 2 do mesmo mercado
        if fams.get(c["market"], 0) >= 2:
            continue
        fams[c["market"]] = fams.get(c["market"], 0) + 1
        res.append((m, c))
    return res[: b.get("safe_count", 5)]


def build_multi(pool: list[dict], cfg: dict) -> dict | None:
    """Múltipla: 2-4 seleções prováveis, de jogos diferentes."""
    b = cfg.get("boletim", {})
    lo, hi = b.get("multi_odds", [2.0, 5.5])
    legs = [c for c in pool if c["p"] >= b.get("multi_min_prob", 0.58) and "reject" not in c]
    legs.sort(key=lambda c: -(c["p"] * (1 + max(c.get("vs_mkt", 0), 0))))
    legs = legs[:8]
    best = None
    for k in (2, 3, 4):
        for combo in itertools.combinations(legs, k):
            if len({c["mid"] for c in combo}) < k:
                continue
            o = math.prod(c["odds"] for c in combo)
            if not (lo <= o <= hi):
                continue
            p = math.prod(c["p"] for c in combo)
            score = p * o - 1 + 0.4 * p
            if best is None or score > best[0]:
                best = (score, combo, o, p)
    if not best:
        return None
    _, combo, o, p = best
    return {"legs": [c["key"] for c in combo], "odds": round(o, 2), "prob": p}


def settle_sel(market: str, code: str, line, hg: int, ag: int):
    """Resultado de uma seleção: (fração ganha, fração devolvida, fração perdida)."""
    gd, tot = hg - ag, hg + ag
    if market == "1x2":
        r = "H" if gd > 0 else ("D" if gd == 0 else "A")
        return (1, 0, 0) if r == code else (0, 0, 1)
    if market == "dc":
        r = "H" if gd > 0 else ("D" if gd == 0 else "A")
        return (1, 0, 0) if r in {"1X": "HD", "X2": "AD", "12": "HA"}[code] else (0, 0, 1)
    if market.startswith("ou"):
        over = tot > float(line)
        return (1, 0, 0) if over == (code == "O") else (0, 0, 1)
    if market == "btts":
        y = hg > 0 and ag > 0
        return (1, 0, 0) if y == (code == "Y") else (0, 0, 1)
    if market.startswith("ah"):
        return markets._ah_parts({gd: 1.0}, float(line) if code == "H" else -float(line),
                                 "home" if code == "H" else "away")
    return (0, 0, 1)


def profit_units(w, p, l, odds: float) -> float:
    return w * (odds - 1) - l
