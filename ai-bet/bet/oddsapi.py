"""The Odds API (the-odds-api.com): odds ao vivo de campeonatos, competições
europeias e taças.

Plano grátis: 500 créditos por mês. Custos:
  /sports e /events                      grátis
  /sports/{k}/odds  (h2h,totals, eu)     2 créditos por competição com jogos
  /events/{id}/odds (btts,spreads)       1 crédito por mercado devolvido, só nos jogos escolhidos
  /scores?daysFrom=3                     2 créditos por competição
O código só pede odds às competições que têm jogos na janela (a lista de jogos
é grátis) e para de gastar quando os créditos ficam abaixo de `reserve`.

Formato da resposta (v4):
  [{"id","sport_key","sport_title","commence_time","home_team","away_team",
    "bookmakers":[{"key","title","markets":[{"key":"h2h","outcomes":[{"name","price","point"?}]}]}]}]
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import names

BASE = "https://api.the-odds-api.com/v4"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "oddsapi"
SECRETS = ROOT / "data" / "secrets.json"

COMPETITIONS = {
    "soccer_portugal_primeira_liga": ("Liga Portugal", "P1", "🇵🇹"),
    "soccer_epl": ("Premier League", "E0", "🏴"),
    "soccer_spain_la_liga": ("La Liga", "SP1", "🇪🇸"),
    "soccer_italy_serie_a": ("Serie A", "I1", "🇮🇹"),
    "soccer_germany_bundesliga": ("Bundesliga", "D1", "🇩🇪"),
    "soccer_france_ligue_one": ("Ligue 1", "F1", "🇫🇷"),
    "soccer_netherlands_eredivisie": ("Eredivisie", "N1", "🇳🇱"),
    "soccer_uefa_champs_league": ("Liga dos Campeões", None, "⭐"),
    "soccer_uefa_europa_league": ("Liga Europa", None, "🟠"),
    "soccer_uefa_europa_conference_league": ("Liga Conferência", None, "🟢"),
    "soccer_fa_cup": ("Taça de Inglaterra", None, "🏆"),
    "soccer_england_efl_cup": ("Taça da Liga Inglesa", None, "🏆"),
    "soccer_spain_copa_del_rey": ("Taça do Rei", None, "🏆"),
    "soccer_italy_coppa_italia": ("Taça de Itália", None, "🏆"),
    "soccer_germany_dfb_pokal": ("Taça da Alemanha", None, "🏆"),
    "soccer_france_coupe_de_france": ("Taça de França", None, "🏆"),
}
DIV_TO_KEY = {v[1]: k for k, v in COMPETITIONS.items() if v[1]}


class Quota:
    remaining: int | None = None
    used: int | None = None
    spent_run = 0


def api_key(cfg: dict | None = None) -> str | None:
    k = os.environ.get("ODDS_API_KEY") or ((cfg or {}).get("odds_api") or {}).get("key")
    if not k and SECRETS.exists():
        try:
            k = json.loads(SECRETS.read_text()).get("odds_api_key")
        except Exception:  # noqa: BLE001
            k = None
    return k or None


def save_key(k: str):
    SECRETS.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(SECRETS.read_text()) if SECRETS.exists() else {}
    d["odds_api_key"] = k.strip()
    SECRETS.write_text(json.dumps(d))
    os.chmod(SECRETS, 0o600)


def _get(path: str, params: dict, key: str, cache_name: str | None = None):
    q = urllib.parse.urlencode({**params, "apiKey": key})
    req = urllib.request.Request(f"{BASE}{path}?{q}", headers={"User-Agent": "ai-bet"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        rem = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        last = r.headers.get("x-requests-last")
    if rem is not None:
        Quota.remaining = int(float(rem))
    if used is not None:
        Quota.used = int(float(used))
    if last is not None:
        Quota.spent_run += int(float(last))
    data = json.loads(body)
    if cache_name:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{cache_name}.json").write_text(json.dumps(data))
    return data


def _cached(name: str, max_age_h: float):
    p = CACHE / f"{name}.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        return json.loads(p.read_text())
    return None


def fetch_window(cfg: dict, hours: float, now_ts: float | None = None, log=print,
                 offline_dir: Path | None = None) -> list[dict]:
    """Jogos (com odds h2h + totals) das competições configuradas nas próximas `hours` horas."""
    oc = cfg.get("odds_api", {})
    sports = oc.get("sports") or list(COMPETITIONS)
    if offline_dir is not None:  # demo: ficheiros no formato da API
        out = []
        for s in sports:
            p = offline_dir / f"odds_{s}.json"
            if p.exists():
                out += json.loads(p.read_text())
        return out
    key = api_key(cfg)
    if not key:
        log("  ⚠️  sem chave da Odds API: só campeonatos do football-data (corre: python -m bet setup-odds)")
        return []
    reserve = oc.get("reserve", 40)
    regions = oc.get("regions", "eu")
    now_ts = now_ts or time.time()
    lim = now_ts + hours * 3600
    try:
        active = {s["key"] for s in _get("/sports", {}, key)}
    except Exception as e:  # noqa: BLE001
        log(f"  ⚠️  Odds API indisponível: {e}")
        return []
    out = []
    for s in sports:
        if s not in active:
            continue
        try:
            evs = _get(f"/sports/{s}/events", {"dateFormat": "unix"}, key)
        except Exception as e:  # noqa: BLE001
            log(f"  ⚠️  {s}: {e}")
            continue
        if not any(now_ts - 3600 < e["commence_time"] < lim for e in evs):
            continue
        if Quota.remaining is not None and Quota.remaining < reserve:
            log(f"  ⚠️  créditos da Odds API quase no fim ({Quota.remaining}); a usar só dados grátis")
            break
        cached = _cached(f"odds_{s}", oc.get("cache_hours", 2))
        if cached is None:
            try:
                cached = _get(f"/sports/{s}/odds", {"regions": regions, "markets": "h2h,totals",
                                                    "oddsFormat": "decimal", "dateFormat": "iso"},
                              key, cache_name=f"odds_{s}")
            except Exception as e:  # noqa: BLE001
                log(f"  ⚠️  odds {s}: {e}")
                continue
        out += cached
    return out


def fetch_extra(cfg: dict, sport: str, event_id: str, log=print, offline_dir: Path | None = None) -> dict | None:
    """Mercados extra (ambas marcam, handicap asiático) para um jogo escolhido."""
    oc = cfg.get("odds_api", {})
    markets = oc.get("extra_markets", "btts,spreads")
    if not markets:
        return None
    if offline_dir is not None:
        p = offline_dir / f"event_{event_id}.json"
        return json.loads(p.read_text()) if p.exists() else None
    key = api_key(cfg)
    if not key or (Quota.remaining is not None and Quota.remaining < oc.get("reserve", 40)):
        return None
    cached = _cached(f"event_{event_id}", oc.get("cache_hours", 2))
    if cached is not None:
        return cached
    try:
        return _get(f"/sports/{sport}/events/{event_id}/odds",
                    {"regions": oc.get("regions", "eu"), "markets": markets, "oddsFormat": "decimal"},
                    key, cache_name=f"event_{event_id}")
    except Exception as e:  # noqa: BLE001
        log(f"  ⚠️  mercados extra {event_id}: {e}")
        return None


def fetch_scores(cfg: dict, sports: list[str], log=print, offline_dir: Path | None = None) -> list[dict]:
    if offline_dir is not None:
        out = []
        for s in sports:
            p = offline_dir / f"scores_{s}.json"
            if p.exists():
                out += json.loads(p.read_text())
        return out
    key = api_key(cfg)
    if not key:
        return []
    out = []
    for s in sports:
        if Quota.remaining is not None and Quota.remaining < cfg.get("odds_api", {}).get("reserve", 40) // 2:
            break
        try:
            out += _get(f"/sports/{s}/scores", {"daysFrom": 3, "dateFormat": "iso"}, key)
        except Exception as e:  # noqa: BLE001
            log(f"  ⚠️  resultados {s}: {e}")
    return out


# ------------------------------------------------------------------ parsing
def _book_key(b):
    return {"pinnacle": "PS", "betfair_ex_eu": "BF", "williamhill": "WH", "betclic": "Betclic",
            "unibet_eu": "Unibet", "marathonbet": "Marathon", "betsson": "Betsson", "sport888": "888",
            "onexbet": "1xBet", "nordicbet": "NordicBet", "coolbet": "Coolbet", "tipico_de": "Tipico",
            "matchbook": "Matchbook", "everygame": "Everygame", "gtbets": "GTbets"}.get(b["key"], b.get("title", b["key"]))


def parse_event(ev: dict, extra: dict | None = None) -> dict:
    """Converte um evento da API em odds por mercado.

    Devolve {"1x2": {casa: [h,d,a]}, "ou": {linha: {casa: [over, under]}},
             "btts": {casa: [sim, não]}, "ah": {linha_casa: {casa: [casa, fora]}}}
    """
    home, away = ev["home_team"], ev["away_team"]
    out = {"1x2": {}, "ou": {}, "btts": {}, "ah": {}}
    books = list(ev.get("bookmakers", []))
    if extra:
        books += extra.get("bookmakers", [])
    for b in books:
        bk = _book_key(b)
        for m in b.get("markets", []):
            oc = m.get("outcomes", [])
            if m["key"] == "h2h":
                d = {o["name"]: o["price"] for o in oc}
                if home in d and away in d and "Draw" in d:
                    out["1x2"][bk] = [d[home], d["Draw"], d[away]]
            elif m["key"] in ("totals", "alternate_totals"):
                lines = {}
                for o in oc:
                    pt = o.get("point")
                    if pt is None:
                        continue
                    lines.setdefault(float(pt), {})[o["name"].lower()] = o["price"]
                for pt, d in lines.items():
                    if "over" in d and "under" in d:
                        out["ou"].setdefault(pt, {})[bk] = [d["over"], d["under"]]
            elif m["key"] == "btts":
                d = {o["name"].lower(): o["price"] for o in oc}
                if "yes" in d and "no" in d:
                    out["btts"][bk] = [d["yes"], d["no"]]
            elif m["key"] in ("spreads", "alternate_spreads"):
                hs = {o.get("point"): o["price"] for o in oc if o["name"] == home}
                as_ = {o.get("point"): o["price"] for o in oc if o["name"] == away}
                for pt, ph in hs.items():
                    if pt is not None and -pt in as_:
                        out["ah"].setdefault(float(pt), {})[bk] = [ph, as_[-pt]]
    return out


def comp_info(sport_key: str):
    return COMPETITIONS.get(sport_key, (sport_key.replace("soccer_", "").replace("_", " ").title(), None, "⚽"))


def find_event(events: list[dict], home: str, away: str, kickoff_ts: float, tol_h: float = 30):
    for e in events:
        t = e["commence_time"]
        if isinstance(t, str):
            from datetime import datetime
            t = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        if abs(t - kickoff_ts) <= tol_h * 3600 and names.key(e["home_team"]) == names.key(home) \
                and names.key(e["away_team"]) == names.key(away):
            return e
    for e in events:
        if names.match(home, [e["home_team"]]) and names.match(away, [e["away_team"]]):
            return e
    return None
