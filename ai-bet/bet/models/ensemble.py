"""Combinação de modelos por pooling logarítmico.

    log p_final(k) = sum_j w_j * log p_j(k) + const

Os pesos w_j >= 0 são aprendidos em previsões fora-da-amostra (walk-forward),
minimizando o log loss. É isto que corrige a calibração (Walsh & Joshi 2024:
para apostar, calibração vale mais do que acerto). Se a soma dos pesos for
maior que 1, as probabilidades são "esticadas" (extremizing), o que é útil
quando os modelos são individualmente tímidos. O mercado entra como mais um
modelo, normalmente com o maior peso: na literatura, as odds de fecho da
Pinnacle são o melhor preditor disponível.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def pool(prob_list: list, weights) -> np.ndarray:
    lp = np.zeros(len(prob_list[0]))
    for p, w in zip(prob_list, weights):
        lp += w * np.log(np.clip(np.asarray(p, float), 1e-9, 1))
    lp -= lp.max()
    e = np.exp(lp)
    return e / e.sum()


class LogPool:
    def __init__(self, names: list[str], default: dict[str, float] | None = None):
        self.names = names
        d = default or {}
        self.w = np.array([d.get(n, 1.0 / len(names)) for n in names])
        self.n_fit = 0
        self.loss = None

    def fit(self, preds: dict[str, np.ndarray], outcome: np.ndarray, min_n: int = 150):
        """preds[name] -> array (N, K); outcome -> índices (N,)."""
        N = len(outcome)
        if N < min_n:
            return self
        logs = np.stack([np.log(np.clip(preds[n], 1e-9, 1)) for n in self.names])  # J,N,K

        def nll(w):
            lp = np.tensordot(w, logs, 1)
            lp = lp - lp.max(1, keepdims=True)
            lse = np.log(np.exp(lp).sum(1))
            return -(lp[np.arange(N), outcome] - lse).mean() + 1e-3 * (w ** 2).sum()

        res = minimize(nll, self.w, method="L-BFGS-B", bounds=[(0, 2.5)] * len(self.names))
        self.w = res.x
        self.n_fit = N
        self.loss = float(res.fun)
        return self

    def predict(self, preds: dict[str, list[float]]) -> np.ndarray:
        names, ws, ps = [], [], []
        for n, w in zip(self.names, self.w):
            if preds.get(n) is not None:
                names.append(n); ws.append(w); ps.append(preds[n])
        if not ps:
            raise ValueError("sem previsões para combinar")
        # se faltar algum modelo, reescala para manter a "temperatura"
        tot = self.w.sum()
        ws = np.array(ws)
        if ws.sum() > 0:
            ws = ws * tot / ws.sum()
        return pool(ps, ws)

    def as_dict(self):
        return {n: round(float(w), 3) for n, w in zip(self.names, self.w)}
