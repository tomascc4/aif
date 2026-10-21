"""OCTO BET — linha de comandos.

python -m bet demo [--fresh] [--cycles N]   demo com ligas inventadas (sem internet, sem quota)
python -m bet setup-odds                    guardar a chave da The Odds API
python -m bet check                         testar fontes de dados e o Claude
python -m bet run                           fazer o boletim do dia
python -m bet dashboard                     abrir o dashboard
python -m bet backtest                      backtest dos modelos estatísticos (sem LLM)
python -m bet reset                         apagar historial e eventos
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
import tomllib
from pathlib import Path

from . import data, db

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path: str | None) -> dict:
    p = Path(path) if path else ROOT / "config.toml"
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("llm", {}).setdefault("backend", "claude")
    cfg["llm"].setdefault("model", "sonnet")
    cfg["llm"].setdefault("pm_model", "opus")
    cfg["llm"].setdefault("scout_model", cfg["llm"]["model"])
    cfg.setdefault("odds_api", {})
    cfg["_name"] = p.stem
    return cfg


def use_demo(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["leagues"] = ["P1", "SP1"]
    cfg["llm"]["backend"] = "mock"
    cfg["llm"]["max_deep_matches"] = 12
    cfg["llm"]["deep_window_hours"] = 96
    cfg["odds_api"]["sports"] = ["soccer_portugal_primeira_liga", "soccer_spain_la_liga", "soccer_uefa_champs_league"]
    data.RAW = ROOT / "data" / "demo" / "raw"
    db.set_path(ROOT / "data" / "bet.demo.db")
    cfg["_demo"] = True
    return cfg


def use_real(cfg: dict) -> dict:
    name = cfg.get("_name", "config")
    db.set_path(ROOT / "data" / ("bet.db" if name == "config" else f"bet.{name}.db"))
    return cfg


def cmd_demo(cfg, a):
    from . import synth, tally
    from .orchestrator import Run
    cfg = use_demo(cfg)
    w = synth.World(ROOT / "data" / "demo")
    if a.fresh or not w.state_p.exists():
        print("🎲 A criar ligas inventadas (4 épocas de histórico + uma 'Liga dos Campeões')...")
        db.reset()
        w.init(past_seasons=4, played_rounds=8)
    else:
        w.load()
    if a.advance_first:
        n = w.advance()
        print(f"🏁 {n} jogos disputados")
    srv = None if a.no_dashboard else _serve(cfg, background=True)
    for i in range(a.cycles):
        print(f"\n⚽ Dia de jogos {i + 1}/{a.cycles} — {w.now:%d/%m/%Y}")
        print("  →", Run(cfg, now=w.now, world=w).go(refresh=False))
        if i < a.cycles - 1 or a.play_last:
            n = w.advance()
            print(f"  🏁 {n} jogos disputados")
    if a.play_last:
        hist = data.load_history(cfg["leagues"], cfg.get("seasons", 4), refresh=False, today=w.now.date())
        Run(cfg, now=w.now, world=w).settle(hist)
    t = tally.refresh()
    print("\n📊 Historial (DEMO, dados inventados):", t["singles"])
    if srv:
        print(f"\nDashboard em http://localhost:{cfg['dashboard']['port']}  (Ctrl+C para sair)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def cmd_setup_odds(cfg, a):
    from . import oddsapi
    print("Cria uma conta grátis em https://the-odds-api.com e copia a API key do email.")
    k = getpass.getpass("Cola aqui a chave (não aparece no ecrã) e carrega Enter: ").strip()
    if len(k) < 20:
        print("❌ Isso não parece uma chave."); return
    oddsapi.save_key(k)
    try:
        sp = oddsapi._get("/sports", {}, k)
        print(f"✅ Chave guardada. {sum(1 for s in sp if s['key'].startswith('soccer'))} competições de futebol ativas"
              + (f" · créditos restantes: {oddsapi.Quota.remaining}" if oddsapi.Quota.remaining is not None else ""))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Guardei a chave mas o teste falhou: {e}")


def cmd_check(cfg, a):
    from . import clubelo, llm, oddsapi
    cfg = use_real(cfg)
    print("1) Campeonatos (football-data.co.uk)")
    p = data.fetch(cfg["leagues"][0], data.recent_seasons(1)[0], current=True)
    print(f"  {'✅' if p and p.exists() else '❌'} {cfg['leagues'][0]} época atual")
    print("2) Odds de todas as competições (The Odds API)")
    k = oddsapi.api_key(cfg)
    if not k:
        print("  ⚠️  sem chave: corre  .venv/bin/python -m bet setup-odds")
    else:
        try:
            sp = oddsapi._get("/sports", {}, k)
            want = set(cfg["odds_api"].get("sports") or [])
            act = [s["key"] for s in sp if s["key"] in want]
            print(f"  ✅ {len(act)} das tuas competições estão ativas · créditos restantes: {oddsapi.Quota.remaining}")
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {e}")
    print("3) Ratings (ClubElo)")
    r = clubelo.ratings()
    print(f"  {'✅' if r else '❌'} {len(r)} clubes")
    print("4) Agentes (Claude)")
    try:
        print(f"  ✅ respondeu: {llm.check(cfg['llm']['backend'], cfg['llm']['model'])}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {e}")


def cmd_run(cfg, a):
    from . import tally
    from .orchestrator import Run
    cfg = use_real(cfg)
    lock = db.kv_get("lock")
    if lock and time.time() - lock < 3 * 3600 and not a.force:
        print("Já há um ciclo a correr (usa --force se tiveres a certeza que não)."); return
    db.kv_set("lock", time.time())
    try:
        print(Run(cfg).go(refresh=True))
        tally.refresh()
    finally:
        db.kv_set("lock", None)


def cmd_backtest(cfg, a):
    from . import backtest
    from .quant import QuantEngine
    cfg = use_real(cfg)
    print("📥 A carregar histórico...")
    hist = data.load_history(cfg["leagues"], max(cfg.get("seasons", 4), a.seasons), refresh=True)
    print(f"   {int(hist['played'].sum())} jogos · 🔁 walk-forward (a primeira vez demora uns minutos)...")
    QuantEngine(cfg).walk_forward(hist)
    rows = db.oos_all()
    print("\nmodelo     apostas  acerto  ROI(1u)  CLV médio")
    last = None
    for st in ("ensemble", "market", "dc", "sot"):
        r = backtest.run(cfg, rows, strategy=st)
        if "error" in r:
            print(st, r["error"]); continue
        last = r if st == "ensemble" else last
        print(f"{st:<10}{r['n_bets']:>7}  {(r['hit_rate'] or 0):>6.0%}  {r['roi']:>+7.2%}  {(r['avg_clv'] or 0):>+7.2%}")
    if last and "scores" in last:
        print("\nLog loss fora-da-amostra (menor = melhor):")
        for k, v in last["scores"].items():
            print(f"  {k:<5} {v.get('logloss')}")
        print("Pesos que o histórico sugere (1X2):", last["weights"]["1x2"])
        print("→ servem de referência para a secção [blend] do config.toml")


def _serve(cfg, background=False):
    from . import server
    return server.serve(cfg, background=background)


def cmd_dashboard(cfg, a):
    cfg = use_demo(cfg) if a.demo else use_real(cfg)
    print(f"Dashboard em http://localhost:{cfg['dashboard']['port']}")
    _serve(cfg)


def cmd_reset(cfg, a):
    use_demo(cfg) if a.demo else use_real(cfg)
    db.reset()
    print("Historial apagado.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bet")
    ap.add_argument("--config", default=None)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo"); d.add_argument("--fresh", action="store_true")
    d.add_argument("--cycles", type=int, default=6); d.add_argument("--no-dashboard", action="store_true")
    d.add_argument("--play-last", action="store_true", help="joga também o último dia (fecha tudo no historial)")
    d.add_argument("--advance-first", action="store_true", help="joga o dia pendente antes de analisar o seguinte")
    sp.add_parser("setup-odds")
    sp.add_parser("check")
    r = sp.add_parser("run"); r.add_argument("--force", action="store_true")
    b = sp.add_parser("backtest"); b.add_argument("--seasons", type=int, default=5)
    ds = sp.add_parser("dashboard"); ds.add_argument("--demo", action="store_true")
    rs = sp.add_parser("reset"); rs.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    {"demo": cmd_demo, "setup-odds": cmd_setup_odds, "check": cmd_check, "run": cmd_run,
     "backtest": cmd_backtest, "dashboard": cmd_dashboard, "reset": cmd_reset}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
