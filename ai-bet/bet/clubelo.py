"""ClubElo (clubelo.com): ratings Elo de clubes europeus, grátis e sem chave.

Serve para comparar equipas de ligas diferentes (Liga dos Campeões, Liga
Europa, Conferência), onde o Dixon-Coles de uma liga não chega.
  http://api.clubelo.com/AAAA-MM-DD  ->  CSV: Rank,Club,Country,Level,Elo,From,To
"""
from __future__ import annotations

import io
import math
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import names

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "clubelo"
HFA = 60.0  # vantagem de jogar em casa, em pontos Elo (ordem de grandeza típica)


def ratings(d: date | None = None, offline: Path | None = None) -> dict[str, float]:
    d = d or date.today()
    if offline is not None:
        p = offline / "clubelo.csv"
        df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / f"{d.isoformat()}.csv"
        if not p.exists() or time.time() - p.stat().st_mtime > 24 * 3600:
            err = None
            for day in (d, d - timedelta(days=1)):
                for scheme in ("http", "https"):
                    try:
                        req = urllib.request.Request(f"{scheme}://api.clubelo.com/{day.isoformat()}",
                                                     headers={"User-Agent": "Mozilla/5.0 (ai-bet)"})
                        with urllib.request.urlopen(req, timeout=25) as r:
                            body = r.read()
                        if body.count(b"\n") > 50:
                            p.write_bytes(body)
                            err = None
                            break
                    except Exception as e:  # noqa: BLE001
                        err = e
                if p.exists() and time.time() - p.stat().st_mtime < 60:
                    break
            if err and not p.exists():
                old = sorted(CACHE.glob("*.csv"))
                if old:
                    print(f"  ⚠️  ClubElo em baixo ({err}); a usar os ratings de {old[-1].stem}")
                    p = old[-1]
                else:
                    print(f"  ⚠️  ClubElo em baixo ({err}); os jogos entre ligas usam só o mercado")
        if not p.exists():
            return {}
        df = pd.read_csv(io.StringIO(p.read_text(errors="ignore")))
    if df.empty or "Club" not in df or "Elo" not in df:
        return {}
    return {str(r.Club): float(r.Elo) for r in df.itertuples() if not pd.isna(r.Elo)}


def lookup(table: dict[str, float], team: str) -> float | None:
    m = names.match(team, table.keys(), cutoff=0.86)
    return table.get(m) if m else None


def expectancy(elo_h: float, elo_a: float, neutral: bool = False) -> float:
    """Pontos esperados da equipa da casa (vitória=1, empate=0.5)."""
    dr = elo_h - elo_a + (0 if neutral else HFA)
    return 1 / (10 ** (-dr / 400) + 1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
