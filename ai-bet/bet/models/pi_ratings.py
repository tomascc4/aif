"""Pi-ratings (Constantinou & Fenton, 2013), com um logit ordenado a converter
o rating em probabilidades 1X2.

Cada equipa tem dois ratings, um em casa e outro fora, medidos em "golos
acima da média". Depois de cada jogo, os dois ratings são corrigidos pelo
erro entre a diferença de golos prevista e a observada. O erro é amortecido
por log, porque ganhar 5-0 em vez de 3-0 diz pouco mais sobre a força real
da equipa. Na literatura bate o Elo para futebol.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


class PiRatings:
    def __init__(self, lr: float = 0.054, gamma: float = 0.79, b: float = 10.0, c: float = 3.0):
        self.lr, self.gamma, self.b, self.c = lr, gamma, b, c
        self.r: dict[str, list[float]] = {}  # team -> [home, away]
        self.beta, self.t1, self.t2 = 1.0, -0.4, 0.4

    def _g(self, r: float) -> float:
        return math.copysign(self.b ** (abs(r) / self.c) - 1, r)

    def rating(self, team):
        return self.r.setdefault(team, [0.0, 0.0])

    def expected_gd(self, home: str, away: str) -> float:
        return self._g(self.rating(home)[0]) - self._g(self.rating(away)[1])

    def update(self, home: str, away: str, hg: int, ag: int) -> float:
        rh, ra = self.rating(home), self.rating(away)
        pred = self.expected_gd(home, away)
        err = (hg - ag) - pred
        psi = math.copysign(self.c * math.log10(1 + abs(err)), err)
        d_home = psi * self.lr
        rh[0] += d_home
        rh[1] += d_home * self.gamma
        ra[1] -= d_home
        ra[0] -= d_home * self.gamma
        return pred

    # ordered logit: P(A)=s(t1-b*x), P(A ou D)=s(t2-b*x)
    def fit_mapping(self, gd_pred, result):
        x = np.asarray(gd_pred, float)
        y = np.asarray(result)  # 0=H 1=D 2=A

        def nll(p):
            b, t1, dt = p
            t2 = t1 + abs(dt)
            pa = 1 / (1 + np.exp(-(t1 - b * x)))
            pad = 1 / (1 + np.exp(-(t2 - b * x)))
            probs = np.stack([1 - pad, pad - pa, pa], 1)
            pr = np.clip(probs[np.arange(len(y)), y], 1e-9, 1)
            return -np.log(pr).sum()

        res = minimize(nll, [1.0, -0.4, 0.8], method="Nelder-Mead", options={"maxiter": 2000})
        b, t1, dt = res.x
        self.beta, self.t1, self.t2 = float(b), float(t1), float(t1 + abs(dt))

    def predict_1x2(self, home: str, away: str) -> list[float]:
        x = self.expected_gd(home, away)
        pa = 1 / (1 + math.exp(-(self.t1 - self.beta * x)))
        pad = 1 / (1 + math.exp(-(self.t2 - self.beta * x)))
        return [1 - pad, pad - pa, pa]
