"""Do modelo de golos para TODOS os mercados.

Um jogo resume-se a dois números, os golos esperados de cada equipa (λ, μ).
Com eles, o Dixon-Coles dá a probabilidade de cada resultado exato e daí sai
qualquer mercado: 1X2, dupla hipótese, mais/menos golos, ambas marcam e
handicap asiático (com reembolso nas linhas inteiras e meias apostas nas
linhas de quarto).

Para cada seleção calcula-se:
  p_win / p_push / p_lose    na NOSSA estimativa
  justa                      odd com valor esperado zero
  mínima                     odd a partir da qual a aposta tem a edge mínima (config)
  p_mercado                  a mesma conta feita com o λ, μ implícitos no mercado
  melhor odd                 a melhor odd real encontrada, quando existe
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from . import odds as oddsmod

RHO = -0.06
G = 11


def matrix(lam: float, mu: float, rho: float = RHO) -> np.ndarray:
    g = np.arange(G)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def implied_rates(p1x2, p_over: float | None = None, line: float = 2.5, rho: float = RHO):
    """λ, μ que reproduzem as probabilidades (sem margem) do mercado."""
    p1x2 = np.asarray(p1x2, float)
    g = np.arange(G)
    tot = g[:, None] + g[None, :]

    def loss(x):
        lam, mu = math.exp(x[0]), math.exp(x[1])
        m = matrix(lam, mu, rho)
        q = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
        e = ((q - p1x2) ** 2).sum()
        if p_over is not None:
            e += 1.5 * (m[tot > line].sum() - p_over) ** 2
        else:  # puxa o total para um valor típico se o mercado não o dá
            e += 0.002 * (lam + mu - 2.6) ** 2
        return e

    r = minimize(loss, [math.log(1.4), math.log(1.1)], method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 600})
    return math.exp(r.x[0]), math.exp(r.x[1])


def rates_from_expectancy(we: float, total: float, rho: float = RHO):
    """λ, μ com soma `total` e pontos esperados da casa = `we` (vindo do Elo)."""
    lo, hi = -3.0, 3.0
    for _ in range(50):
        s = (lo + hi) / 2
        lam = total * math.exp(s) / (1 + math.exp(s))
        mu = total - lam
        m = matrix(lam, mu, rho)
        e = np.tril(m, -1).sum() + 0.5 * np.trace(m)
        if e < we:
            lo = s
        else:
            hi = s
    return lam, mu


# ------------------------------------------------------------ settlement
def _gd_probs(m: np.ndarray) -> dict[int, float]:
    out = {}
    n = m.shape[0]
    for i in range(n):
        for j in range(n):
            out[i - j] = out.get(i - j, 0.0) + m[i, j]
    return out


def _ah_parts(gd: dict, line: float, side: str):
    """(win, push, lose) para handicap asiático; side 'home' ou 'away'. Trata linhas de quarto."""
    q = round(line * 4) % 2 == 1  # .25 / .75
    lines = [line - 0.25, line + 0.25] if q else [line]
    w = p = l = 0.0
    for ln in lines:
        for d, pr in gd.items():
            m = (d + ln) if side == "home" else (-d + ln)
            if m > 1e-9:
                w += pr
            elif m < -1e-9:
                l += pr
            else:
                p += pr
    k = len(lines)
    return w / k, p / k, l / k


def selections(m: np.ndarray, home: str, away: str, ah_lines=None) -> list[dict]:
    """Todas as seleções que sabemos avaliar, com (win, push, lose)."""
    n = m.shape[0]
    g = np.arange(n)
    tot = g[:, None] + g[None, :]
    ph, pd_, pa = float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())
    out = [
        _s("1x2", "H", f"{home} vence", ph), _s("1x2", "D", "Empate", pd_), _s("1x2", "A", f"{away} vence", pa),
        _s("dc", "1X", f"{home} ou empate", ph + pd_), _s("dc", "X2", f"{away} ou empate", pa + pd_),
        _s("dc", "12", "Não há empate", ph + pa),
    ]
    for ln in (1.5, 2.5, 3.5):
        po = float(m[tot > ln].sum())
        out.append(_s(f"ou{ln}", "O", f"Mais de {ln:g} golos".replace(".", ","), po, line=ln))
        out.append(_s(f"ou{ln}", "U", f"Menos de {ln:g} golos".replace(".", ","), 1 - po, line=ln))
    pb = float(m[1:, 1:].sum())
    out.append(_s("btts", "Y", "Ambas marcam: Sim", pb))
    out.append(_s("btts", "N", "Ambas marcam: Não", 1 - pb))
    gd = _gd_probs(m)
    for ln in (ah_lines or (-2.5, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5)):
        w, p, l = _ah_parts(gd, ln, "home")
        out.append({"market": f"ah{ln:+g}", "code": "H", "label": f"{home} {_fmt_line(ln)} (HA)", "line": ln,
                    "win": w, "push": p, "lose": l})
        w, p, l = _ah_parts(gd, -ln, "away")
        out.append({"market": f"ah{ln:+g}", "code": "A", "label": f"{away} {_fmt_line(-ln)} (HA)", "line": ln,
                    "win": w, "push": p, "lose": l})
    return out


def _fmt_line(x):
    s = f"{x:+g}".replace(".", ",")
    return "0" if x == 0 else s


def _s(market, code, label, p, line=None):
    return {"market": market, "code": code, "label": label, "line": line, "win": p, "push": 0.0, "lose": 1 - p}


def fair_odds(s) -> float:
    return 1 + s["lose"] / max(s["win"], 1e-9)


def min_odds(s, edge: float) -> float:
    return 1 + (edge + s["lose"]) / max(s["win"], 1e-9)


def ev(s, o: float) -> float:
    return s["win"] * (o - 1) - s["lose"]


def eff_prob(s) -> float:
    """Probabilidade "de ganhar", ignorando reembolsos (para mostrar)."""
    den = s["win"] + s["lose"]
    return s["win"] / den if den > 0 else 0.0


# --------------------------------------------------- odds reais por seleção
def market_odds(sel: dict, book_odds: dict) -> dict[str, float]:
    """As odds reais que existem para esta seleção, por casa."""
    mk, c = sel["market"], sel["code"]
    if mk == "1x2":
        i = "HDA".index(c)
        return {b: o[i] for b, o in book_odds.get("1x2", {}).items()}
    if mk.startswith("ou"):
        d = book_odds.get("ou", {}).get(sel["line"], {})
        i = 0 if c == "O" else 1
        return {b: o[i] for b, o in d.items()}
    if mk == "btts":
        i = 0 if c == "Y" else 1
        return {b: o[i] for b, o in book_odds.get("btts", {}).items()}
    if mk.startswith("ah"):
        d = book_odds.get("ah", {}).get(sel["line"], {})
        i = 0 if c == "H" else 1
        return {b: o[i] for b, o in d.items()}
    return {}


def consensus_two_way(book_odds: dict[str, list], sharp="PS"):
    """Probabilidades sem margem de um mercado de 2 vias (consenso)."""
    return oddsmod.consensus(book_odds, "shin", sharp=sharp)
