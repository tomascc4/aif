"""Dados do football-data.co.uk: resultados, remates, árbitros e odds de
várias casas, incluindo as odds de FECHO, que são a chave para medir o CLV.

Grátis e sem API key:
  - https://www.football-data.co.uk/mmz4281/{2526}/{P1}.csv -> época completa
  - https://www.football-data.co.uk/fixtures.csv             -> próximos jogos com odds

As odds "normais" (B365H, PSH, ...) são recolhidas antes da jornada (em
geral à sexta para o fim de semana e à terça para jogos a meio da semana).
As odds de fecho (B365CH, PSCH, ...) são as últimas antes do apito inicial.
"""
from __future__ import annotations

import io
import math
import os
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://www.football-data.co.uk"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

LEAGUES = {
    "P1": "Liga Portugal", "E0": "Premier League", "E1": "Championship",
    "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1",
    "N1": "Eredivisie", "B1": "Jupiler Pro League", "SC0": "Scottish Premiership",
    "T1": "Süper Lig", "G1": "Super League Grécia", "SP2": "La Liga 2", "I2": "Serie B",
    "D2": "2. Bundesliga", "F2": "Ligue 2", "E2": "League One",
}

# casa -> prefixo das colunas 1X2 (abertura, fecho)
BOOKS_1X2 = {
    "B365": ("B365", "B365C"), "BW": ("BW", "BWC"), "IW": ("IW", "IWC"),
    "PS": ("PS", "PSC"), "WH": ("WH", "WHC"), "VC": ("VC", "VCC"),
    "BF": ("BFE", "BFEC"), "Avg": ("Avg", "AvgC"), "Max": ("Max", "MaxC"),
}
# casa -> prefixo das colunas Over/Under 2.5 (abertura, fecho)
BOOKS_OU = {
    "B365": ("B365", "B365C"), "PS": ("P", "PC"), "Avg": ("Avg", "AvgC"),
    "Max": ("Max", "MaxC"), "BF": ("BFE", "BFEC"),
}
AGGREGATES = {"Avg", "Max"}


def season_code(d: date) -> str:
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def recent_seasons(n: int, today: date | None = None) -> list[str]:
    today = today or date.today()
    y = today.year if today.month >= 7 else today.year - 1
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(n)][::-1]


def _download(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ai-bet research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(league: str, season: str, max_age_h: float = 6.0, current: bool = False) -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{league}_{season}.csv"
    fresh = p.exists() and (not current or time.time() - p.stat().st_mtime < max_age_h * 3600)
    if fresh:
        return p
    try:
        data = _download(f"{BASE}/mmz4281/{season}/{league}.csv")
        if len(data) > 200:
            p.write_bytes(data)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  {league} {season}: {e}")
    return p if p.exists() else None


def fetch_fixtures(max_age_h: float = 3.0) -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / "fixtures.csv"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        return p
    try:
        p.write_bytes(_download(f"{BASE}/fixtures.csv"))
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  fixtures: {e}")
    return p if p.exists() else None


# ------------------------------------------------------------------ parsing
def _read_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(txt), on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df.dropna(how="all")


def _f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _odds(row, prefix: str, suffixes) -> list | None:
    vals = [_f(row.get(prefix + s)) for s in suffixes]
    if any(v is None or v <= 1.0 for v in vals):
        return None
    return vals


def _parse_date(s) -> pd.Timestamp | None:
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return None


def normalize(df: pd.DataFrame, league_hint: str | None = None) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        r = r.to_dict()
        d = _parse_date(r.get("Date"))
        if d is None or not r.get("HomeTeam") or not r.get("AwayTeam"):
            continue
        t = str(r.get("Time") or "").strip()
        if len(t) >= 4 and ":" in t:
            hh, mm = t.split(":")[:2]
            try:
                d = d + pd.Timedelta(hours=int(hh), minutes=int(mm))
            except ValueError:
                pass
        hg, ag = _f(r.get("FTHG")), _f(r.get("FTAG"))
        o1, c1, ou, cou = {}, {}, {}, {}
        for b, (po, pc) in BOOKS_1X2.items():
            if (x := _odds(r, po, "HDA")):
                o1[b] = x
            if (x := _odds(r, pc, "HDA")):
                c1[b] = x
        for b, (po, pc) in BOOKS_OU.items():
            if (x := _odds(r, po, (">2.5", "<2.5"))):
                ou[b] = x
            if (x := _odds(r, pc, (">2.5", "<2.5"))):
                cou[b] = x
        rows.append({
            "div": str(r.get("Div") or league_hint or "").strip(),
            "date": d,
            "home": str(r["HomeTeam"]).strip(),
            "away": str(r["AwayTeam"]).strip(),
            "hg": hg, "ag": ag,
            "hst": _f(r.get("HST")), "ast": _f(r.get("AST")),
            "hs": _f(r.get("HS")), "as_": _f(r.get("AS")),
            "referee": (str(r.get("Referee")).strip() if r.get("Referee") and str(r.get("Referee")) != "nan" else None),
            "o1x2": o1, "c1x2": c1, "oou": ou, "cou": cou,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["played"] = out["hg"].notna() & out["ag"].notna()
    out["res"] = np.where(out["hg"] > out["ag"], 0, np.where(out["hg"] == out["ag"], 1, 2))
    out["mid"] = (out["div"] + "|" + out["date"].dt.strftime("%Y-%m-%d") + "|" + out["home"] + "|" + out["away"])
    return out.sort_values("date").reset_index(drop=True)


def load_history(leagues: list[str], seasons: int = 4, refresh: bool = True,
                 today: date | None = None) -> pd.DataFrame:
    frames = []
    ss = recent_seasons(seasons, today)
    for lg in leagues:
        for i, s in enumerate(ss):
            p = fetch(lg, s, current=(i == len(ss) - 1)) if refresh else RAW / f"{lg}_{s}.csv"
            if p and Path(p).exists():
                try:
                    n = normalize(_read_csv(Path(p)), lg)
                    n["season"] = s
                    frames.append(n)
                except Exception as e:  # noqa: BLE001
                    print(f"  ⚠️  erro a ler {p}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("mid", keep="last").sort_values("date").reset_index(drop=True)
    return df


def load_fixtures(leagues: list[str], days_ahead: int = 4, refresh: bool = True,
                  now: datetime | None = None) -> pd.DataFrame:
    p = fetch_fixtures() if refresh else RAW / "fixtures.csv"
    if not p or not Path(p).exists():
        return pd.DataFrame()
    df = normalize(_read_csv(Path(p)))
    if df.empty:
        return df
    now = pd.Timestamp(now or datetime.now())
    df = df[df["div"].isin(leagues)]
    df = df[(df["date"] >= now - pd.Timedelta(hours=2)) & (df["date"] <= now + pd.Timedelta(days=days_ahead))]
    return df.reset_index(drop=True)


def is_demo_data() -> bool:
    return (RAW / "DEMO").exists()
