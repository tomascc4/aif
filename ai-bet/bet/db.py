"""SQLite: eventos dos agentes, previsões fora-da-amostra, jogos analisados,
apostas (paper), banca e estado."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "bet.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, agent TEXT,
    kind TEXT, mid TEXT, text TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS oos (mid TEXT PRIMARY KEY, date TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS matches (
    mid TEXT, run_id TEXT, ts REAL, date TEXT, div TEXT, home TEXT, away TEXT,
    stage TEXT, data TEXT, PRIMARY KEY (mid, run_id));
CREATE TABLE IF NOT EXISTS picks (
    key TEXT PRIMARY KEY, run_id TEXT, ts REAL, kind TEXT, mid TEXT, sport TEXT, comp TEXT, flag TEXT,
    date TEXT, home TEXT, away TEXT, market TEXT, code TEXT, line REAL, label TEXT, p REAL, p_mkt REAL,
    fair REAL, min_odds REAL, odds REAL, book TEXT, estimated INTEGER, edge REAL, stars INTEGER, why TEXT,
    status TEXT DEFAULT 'open', result TEXT, units REAL, clv REAL);
CREATE TABLE IF NOT EXISTS boletins (run_id TEXT PRIMARY KEY, ts REAL, data TEXT);
CREATE TABLE IF NOT EXISTS layer_preds (
    mid TEXT, layer TEXT, market TEXT, p TEXT, res INTEGER, PRIMARY KEY (mid, layer, market));
"""


def set_path(p: Path):
    global DB_PATH
    DB_PATH = Path(p)


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def kv_get(k, default=None):
    with conn() as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r["v"]) if r else default


def kv_set(k, v):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)", (k, json.dumps(v)))


def event(run_id, agent, kind, text, mid=None, data=None):
    with conn() as c:
        c.execute("INSERT INTO events(ts,run_id,agent,kind,mid,text,data) VALUES (?,?,?,?,?,?,?)",
                  (time.time(), run_id, agent, kind, mid, text, json.dumps(data) if data is not None else None))


def events_since(i: int, limit: int = 400):
    with conn() as c:
        rows = c.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (i, limit)).fetchall()
    return [dict(r) | {"data": json.loads(r["data"]) if r["data"] else None} for r in rows]


def oos_ids() -> set:
    with conn() as c:
        return {r[0] for r in c.execute("SELECT mid FROM oos")}


def oos_put(rows):
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO oos(mid,date,data) VALUES (?,?,?)",
                      [(r["mid"], r["date"], json.dumps(r)) for r in rows])


def oos_all():
    with conn() as c:
        return [json.loads(r[0]) for r in c.execute("SELECT data FROM oos ORDER BY date")]


def match_put(run_id, m: dict, stage: str):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO matches(mid,run_id,ts,date,div,home,away,stage,data) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (m["mid"], run_id, time.time(), m["date"], m["div"], m["home"], m["away"], stage,
                   json.dumps(m, default=str)))


def matches_for_run(run_id):
    with conn() as c:
        rows = c.execute("SELECT data, stage FROM matches WHERE run_id=? ORDER BY date", (run_id,)).fetchall()
    return [json.loads(r["data"]) | {"stage": r["stage"]} for r in rows]


def latest_match(mid):
    with conn() as c:
        r = c.execute("SELECT data FROM matches WHERE mid=? ORDER BY ts DESC LIMIT 1", (mid,)).fetchone()
    return json.loads(r[0]) if r else None


PICK_COLS = ["key", "run_id", "ts", "kind", "mid", "sport", "comp", "flag", "date", "home", "away", "market",
             "code", "line", "label", "p", "p_mkt", "fair", "min_odds", "odds", "book", "estimated", "edge",
             "stars", "why"]


def pick_put(p: dict):
    """Insere ou atualiza uma escolha (a mesma seleção no mesmo jogo é uma só linha)."""
    p = {"ts": time.time(), **p}
    with conn() as c:
        old = c.execute("SELECT status, kind FROM picks WHERE key=?", (p["key"],)).fetchone()
        if old and old["status"] != "open":
            return
        if old and old["kind"] == "single" and p.get("kind") == "leg":
            return
        c.execute(f"INSERT OR REPLACE INTO picks({','.join(PICK_COLS)}) VALUES ({','.join('?' * len(PICK_COLS))})",
                  [p.get(k) for k in PICK_COLS])


def picks(status: str | None = None, limit: int = 1000):
    q, a = "SELECT * FROM picks", []
    if status:
        q += " WHERE status=?"; a.append(status)
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY date DESC LIMIT ?", a + [limit])]


def pick_settle(key, status, result, units, clv=None):
    with conn() as c:
        c.execute("UPDATE picks SET status=?, result=?, units=?, clv=? WHERE key=?", (status, result, units, clv, key))


def boletim_put(run_id, data: dict):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO boletins(run_id,ts,data) VALUES (?,?,?)", (run_id, time.time(), json.dumps(data, default=str)))


def boletins(limit=30):
    with conn() as c:
        return [json.loads(r[0]) for r in c.execute("SELECT data FROM boletins ORDER BY ts DESC LIMIT ?", (limit,))]


def boletim_update(run_id, data):
    boletim_put(run_id, data)


def layer_put(mid, layer, market, p, res=None):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO layer_preds(mid,layer,market,p,res) VALUES (?,?,?,?,?)",
                  (mid, layer, market, json.dumps(p), res))


def layer_set_result(mid, market, res):
    with conn() as c:
        c.execute("UPDATE layer_preds SET res=? WHERE mid=? AND market=?", (res, mid, market))


def layer_all():
    with conn() as c:
        return [dict(r) | {"p": json.loads(r["p"])} for r in c.execute("SELECT * FROM layer_preds")]


def reset():
    if DB_PATH.exists():
        DB_PATH.unlink()
