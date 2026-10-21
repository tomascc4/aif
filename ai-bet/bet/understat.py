"""Understat (understat.com): xG (golos esperados) jogo a jogo das 5 grandes ligas.

O xG mede a qualidade das oportunidades criadas e concedidas e prevê os golos
futuros melhor do que os próprios golos, que têm muita sorte à mistura.
Endpoint usado pelo próprio site (JSON):
    GET https://understat.com/getLeagueData/{EPL|La_liga|Bundesliga|Serie_A|Ligue_1}/{ano_inicio_epoca}
    -> {"dates": [{"isResult", "h": {"title"}, "a": {"title"}, "goals": {"h","a"}, "xG": {"h","a"}, "datetime"}], "teams": ..., "players": ...}
Se o Understat estiver em baixo, o sistema continua sem esta fonte.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "understat"
SLUG = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _fetch(slug: str, year: int, max_age_h: float) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{slug}_{year}.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        return json.loads(p.read_text())
    try:
        req = urllib.request.Request(f"https://understat.com/getLeagueData/{slug}/{year}", headers={
            "User-Agent": UA, "X-Requested-With": "XMLHttpRequest", "Accept": "application/json",
            "Referer": f"https://understat.com/league/{slug}/{year}"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.loads(r.read())
        if isinstance(data, dict) and data.get("dates"):
            p.write_text(json.dumps({"dates": data["dates"]}))
            return data
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  Understat {slug} {year}: {str(e)[:120]}")
    return json.loads(p.read_text()) if p.exists() else None


def load(div: str, seasons: int = 3, today: date | None = None, offline: Path | None = None) -> pd.DataFrame:
    """Jogos com xG de uma liga (épocas recentes). Vazio se a liga não existir no Understat."""
    if offline is not None:
        f = offline / f"understat_{div}.json"
        dates = json.loads(f.read_text())["dates"] if f.exists() else []
    else:
        slug = SLUG.get(div)
        if not slug:
            return pd.DataFrame()
        today = today or date.today()
        y0 = today.year if today.month >= 7 else today.year - 1
        dates = []
        for k in range(seasons):
            d = _fetch(slug, y0 - k, 6 if k == 0 else 24 * 30)
            if d:
                dates += d.get("dates", [])
    rows = []
    for m in dates:
        try:
            if not m.get("isResult"):
                continue
            rows.append({"date": pd.Timestamp(m["datetime"]), "home": m["h"]["title"], "away": m["a"]["title"],
                         "xg_h": float(m["xG"]["h"]), "xg_a": float(m["xG"]["a"]),
                         "hg": int(m["goals"]["h"]), "ag": int(m["goals"]["a"])})
        except (KeyError, TypeError, ValueError):
            continue
    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True) if not df.empty else df
