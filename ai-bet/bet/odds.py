"""Odds -> probabilidades justas (sem a margem da casa).

Métodos (Štrumbelj 2014 compara-os; Shin e power tratam melhor o
favourite-longshot bias do que a normalização simples):
    multiplicative  p_i = q_i / sum(q)
    power           p_i = q_i^k, com k tal que sum = 1
    shin            modelo de Shin (1993) com fração z de apostadores informados

Também: consenso entre casas (Kaunitz et al. 2017 — "beating the bookies with
their own numbers"), melhor preço disponível e cálculo de CLV.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def implied(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    return 1.0 / o


def margin(odds) -> float:
    return float(implied(odds).sum() - 1.0)


def devig_multiplicative(odds) -> np.ndarray:
    q = implied(odds)
    return q / q.sum()


def devig_power(odds) -> np.ndarray:
    q = implied(odds)
    if abs(q.sum() - 1) < 1e-9:
        return q
    f = lambda k: (q ** k).sum() - 1.0
    try:
        k = brentq(f, 0.5, 3.0)
    except ValueError:
        return q / q.sum()
    p = q ** k
    return p / p.sum()


def devig_shin(odds) -> np.ndarray:
    q = implied(odds)
    s = q.sum()
    if s <= 1.0 + 1e-9:
        return q / s

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z))

    f = lambda z: probs(z).sum() - 1.0
    try:
        z = brentq(f, 0.0, 0.4)
    except ValueError:
        return devig_power(odds)
    p = probs(z)
    return p / p.sum()


METHODS = {"multiplicative": devig_multiplicative, "power": devig_power, "shin": devig_shin}


def devig(odds, method: str = "shin") -> np.ndarray:
    return METHODS[method](odds)


def valid(odds) -> bool:
    try:
        return all(o is not None and not math.isnan(o) and o > 1.0 for o in odds)
    except TypeError:
        return False


def consensus(book_odds: dict[str, list[float]], method: str = "shin",
              sharp: str | None = "PS", sharp_weight: float = 0.6) -> np.ndarray | None:
    """Probabilidade de consenso do mercado.

    Média das probabilidades justas de cada casa, com peso extra na casa
    "sharp" (Pinnacle), que é a referência de eficiência na literatura.
    """
    ps = {b: devig(o, method) for b, o in book_odds.items() if valid(o)}
    if not ps:
        return None
    if sharp and sharp in ps and len(ps) > 1:
        others = np.mean([p for b, p in ps.items() if b != sharp], axis=0)
        p = sharp_weight * ps[sharp] + (1 - sharp_weight) * others
    else:
        p = np.mean(list(ps.values()), axis=0)
    return p / p.sum()


def best_price(book_odds: dict[str, list[float]], allowed: list[str] | None = None):
    """Melhor odd por resultado entre as casas permitidas -> (odds, casas)."""
    n = None
    best, who = None, None
    for b, o in book_odds.items():
        if allowed and b not in allowed:
            continue
        if not valid(o):
            continue
        if best is None:
            n = len(o)
            best, who = list(o), [b] * n
            continue
        for i in range(n):
            if o[i] > best[i]:
                best[i], who[i] = o[i], b
    return best, who


def clv(taken_odds: float, closing_fair_prob: float) -> float:
    """CLV = valor esperado da aposta medido contra o fecho justo.

    taken_odds * p_fecho - 1. Positivo => batemos o mercado de fecho.
    """
    return taken_odds * closing_fair_prob - 1.0
