"""Servidor do dashboard (só biblioteca standard)."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import db

WEB = Path(__file__).resolve().parent / "web"
ROOT = Path(__file__).resolve().parent.parent

AGENTS = [
    # id, nome, o que faz, grupo
    ("arquivista", "Arquivista", "Resultados, remates e odds de fecho das ligas", "dados"),
    ("corretor", "Corretor", "Odds de várias casas: 1X2, golos, ambas marcam, handicap", "dados"),
    ("elo", "ClubElo", "Força dos clubes entre ligas diferentes", "dados"),
    ("modelos", "Modelos", "Dixon-Coles, remates enquadrados, Elo e mercado → golos esperados", "modelos"),
    ("radar", "Radar", "Varre as notícias de todos os jogos de cada competição", "noticias"),
    ("triagem", "Triagem", "Escolhe os jogos que merecem análise a fundo", "modelos"),
    ("medico", "Médico", "Lesões, castigos e onze provável", "noticias"),
    ("reporter", "Repórter", "Motivação, rotação, treinador, tempo", "noticias"),
    ("analista", "Analista", "Notícias → ajuste dos golos esperados (3 palpites)", "noticias"),
    ("cetico", "Advogado do Diabo", "Tenta deitar abaixo cada aposta", "decisao"),
    ("presidente", "Presidente", "Fecha o boletim e explica cada escolha", "decisao"),
]


def state(cfg) -> dict:
    with db.conn() as c:
        last_ev = c.execute("SELECT MAX(id) FROM events").fetchone()[0] or 0
    hist = [p for p in db.picks(limit=600) if p["kind"] != "leg"]
    return {
        "demo": bool(cfg.get("_demo")), "run": db.kv_get("current_run", {}) or {},
        "boletim": db.kv_get("boletim"), "matches": db.kv_get("matches", []) or [],
        "history": hist, "boletins": db.boletins(20), "tally": db.kv_get("tally"),
        "last_event": last_ev,
        "agents": [dict(zip(("id", "name", "role", "group"), a)) for a in AGENTS],
        "config": {"min_edge": cfg.get("boletim", {}).get("min_edge", 0.03)},
    }


def make_handler(cfg):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(WEB), **k)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/api/state":
                return self._json(state(cfg))
            if u.path == "/api/events":
                return self._json(db.events_since(int(q.get("since", ["0"])[0]), int(q.get("limit", ["400"])[0])))
            return super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path == "/api/run":
                if db.kv_get("lock"):
                    return self._json({"ok": False, "msg": "já está a correr"}, 409)
                args = [sys.executable, "-m", "bet"]
                if cfg.get("_name") and cfg["_name"] != "config":
                    args += ["--config", f"{cfg['_name']}.toml"]
                args += ["demo", "--cycles", "1", "--no-dashboard", "--advance-first"] if cfg.get("_demo") else ["run"]
                subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self._json({"ok": True})
            self._json({"error": "?"}, 404)

    return H


def serve(cfg, background=False):
    port = cfg.get("dashboard", {}).get("port", 8790)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(cfg))
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
