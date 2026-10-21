"""Dixon & Coles (1997): Poisson bivariado com correção de resultados baixos
e decaimento temporal (jogos antigos pesam menos).

    golos_casa ~ Poisson(lambda),  log lambda = ataque[casa] + defesa[fora] + vantagem_casa
    golos_fora ~ Poisson(mu),      log mu     = ataque[fora] + defesa[casa]
    peso do jogo = exp(-xi * dias_desde_o_jogo)

Gradiente analítico para o ajuste ser rápido (o backtest reajusta o modelo
todas as semanas). Uma pequena penalização L2 (ridge) estabiliza equipas com
poucos jogos, como as que sobem de divisão.

`target="sot"` ajusta o mesmo modelo a "golos sintéticos" que misturam golos
e remates enquadrados. Os remates são muito mais numerosos do que os golos e,
por isso, medem a força de uma equipa com menos ruído. É um proxy grátis de
xG, já que o football-data traz HST/AST.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


class DixonColes:
    def __init__(self, xi: float = 0.0019, ridge: float = 0.02, use_rho: bool = True):
        self.xi = xi
        self.ridge = ridge
        self.use_rho = use_rho
        self.teams: list[str] = []
        self.idx: dict[str, int] = {}
        self.att = self.dfn = None
        self.home = 0.25
        self.rho = 0.0
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, home, away, hg, ag, days_ago):
        home = list(home); away = list(away)
        hg = np.asarray(hg, float); ag = np.asarray(ag, float)
        w = np.exp(-self.xi * np.asarray(days_ago, float))
        self.teams = sorted(set(home) | set(away))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = np.array([self.idx[t] for t in home])
        ai = np.array([self.idx[t] for t in away])
        integer = np.allclose(hg, np.round(hg)) and np.allclose(ag, np.round(ag))
        fit_rho = self.use_rho and integer

        c00 = (hg == 0) & (ag == 0)
        c01 = (hg == 0) & (ag == 1)
        c10 = (hg == 1) & (ag == 0)
        c11 = (hg == 1) & (ag == 1)

        def unpack(x):
            return x[:n], x[n:2 * n], x[2 * n], (x[2 * n + 1] if fit_rho else 0.0)

        def nll(x):
            att, dfn, h, rho = unpack(x)
            e1 = att[hi] + dfn[ai] + h
            e2 = att[ai] + dfn[hi]
            lam, mu = np.exp(e1), np.exp(e2)
            ll = hg * e1 - lam + ag * e2 - mu
            g1 = hg - lam
            g2 = ag - mu
            grho = np.zeros_like(lam)
            if fit_rho:
                tau = np.ones_like(lam)
                t00 = 1 - lam * mu * rho
                t01 = 1 + lam * rho
                t10 = 1 + mu * rho
                t11 = 1 - rho
                tau[c00] = t00[c00]; tau[c01] = t01[c01]; tau[c10] = t10[c10]
                tau[c11] = t11
                tau = np.clip(tau, 1e-10, None)
                ll = ll + np.log(tau)
                g1 = g1 + np.where(c00, -lam * mu * rho / np.clip(t00, 1e-10, None), 0)
                g1 = g1 + np.where(c01, lam * rho / np.clip(t01, 1e-10, None), 0)
                g2 = g2 + np.where(c00, -lam * mu * rho / np.clip(t00, 1e-10, None), 0)
                g2 = g2 + np.where(c10, mu * rho / np.clip(t10, 1e-10, None), 0)
                grho = (np.where(c00, -lam * mu / np.clip(t00, 1e-10, None), 0)
                        + np.where(c01, lam / np.clip(t01, 1e-10, None), 0)
                        + np.where(c10, mu / np.clip(t10, 1e-10, None), 0)
                        + np.where(c11, -1 / max(t11, 1e-10), 0))
            f = -(w * ll).sum()
            wg1, wg2 = w * g1, w * g2
            gatt = -(np.bincount(hi, wg1, n) + np.bincount(ai, wg2, n))
            gdef = -(np.bincount(ai, wg1, n) + np.bincount(hi, wg2, n))
            gh = -wg1.sum()
            # identificabilidade: soma dos ataques = 0
            s = att.sum()
            f += 10.0 * s * s
            gatt += 20.0 * s
            # ridge
            f += self.ridge * w.sum() / n * ((att ** 2).sum() + (dfn ** 2).sum()) / 10
            k = self.ridge * w.sum() / n / 10 * 2
            gatt += k * att
            gdef += k * dfn
            grad = np.concatenate([gatt, gdef, [gh]] + ([[-(w * grho).sum()]] if fit_rho else []))
            return f, grad

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25]] + ([[-0.05]] if fit_rho else []))
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1.5)] + ([(-0.3, 0.3)] if fit_rho else [])
        res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        att, dfn, h, rho = unpack(res.x)
        self.att, self.dfn, self.home, self.rho = att, dfn, float(h), float(rho)
        self.fitted = True
        return self

    # -------------------------------------------------------------- predict
    def rates(self, home: str, away: str):
        mean_att = 0.0
        a_h = self.att[self.idx[home]] if home in self.idx else mean_att - 0.15
        d_h = self.dfn[self.idx[home]] if home in self.idx else np.mean(self.dfn) + 0.15
        a_a = self.att[self.idx[away]] if away in self.idx else mean_att - 0.15
        d_a = self.dfn[self.idx[away]] if away in self.idx else np.mean(self.dfn) + 0.15
        lam = float(np.exp(a_h + d_a + self.home))
        mu = float(np.exp(a_a + d_h))
        return lam, mu

    def knows(self, team: str) -> bool:
        return team in self.idx

    def score_matrix(self, lam: float, mu: float, rho: float | None = None) -> np.ndarray:
        rho = self.rho if rho is None else rho
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        m[0, 0] *= 1 - lam * mu * rho
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def predict(self, home: str, away: str, lam_mult: float = 1.0, mu_mult: float = 1.0) -> dict:
        lam, mu = self.rates(home, away)
        lam *= lam_mult
        mu *= mu_mult
        return markets_from_matrix(self.score_matrix(lam, mu), lam, mu)


def markets_from_matrix(m: np.ndarray, lam: float, mu: float) -> dict:
    hw = float(np.tril(m, -1).sum())
    dr = float(np.trace(m))
    aw = float(np.triu(m, 1).sum())
    g = np.arange(m.shape[0])
    tot = g[:, None] + g[None, :]
    over = float(m[tot > 2.5].sum())
    btts = float(m[1:, 1:].sum())
    top = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:6], m.shape))[0]
    return {
        "1x2": [hw, dr, aw],
        "ou25": [over, 1 - over],
        "btts": btts,
        "xg": [lam, mu],
        "top_scores": [[int(i), int(j), float(m[i, j])] for i, j in top],
        "matrix": m[:6, :6].round(4).tolist(),
    }
