"""Mundo sintético para o modo demo.

Gera ligas fictícias com CSVs no MESMO formato do football-data.co.uk, por
isso o pipeline real corre sem alterações. Também simula um mercado de
apostas com as propriedades que a literatura descreve:

  - a "verdade" é um Dixon-Coles com forças que mudam ao longo da época;
  - a Pinnacle (PS) tem pouca margem e pouco ruído, e o fecho é mais preciso
    do que a abertura;
  - as casas "soft" têm mais margem, mais ruído e favourite-longshot bias;
  - alguns jogos recebem notícias de última hora (lesões, rotação) que a
    abertura ainda não reflete mas o fecho já reflete em parte. É isso que dá
    trabalho útil aos batedores.

Nada aqui é real. Os nomes das equipas são inventados.
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

TEAMS = {
    "P1": ["Atlético Azeitão", "Arrábida FC", "Sesimbra United", "CD Palmela", "Setúbal Real",
           "Tróia Marítimo", "Barreiro Ferroviário", "Moita Clube", "Seixal Académico", "Almada SC",
           "Montijo Desportivo", "Alcochete Salinas", "Sintra Serra", "Cascais Atlântico",
           "Oeiras Tejo", "Loures Rangers", "Mafra Convento", "Ericeira Surf"],
    "SP1": ["Real Brisa", "Deportivo Faro", "Atlético Sierra", "CF Olivar", "Unión Costa",
            "Racing Puerto", "Sporting Meseta", "CD Naranjo", "Real Viento", "Club Alcázar",
            "Celta Norte", "Getafuera", "Valle Verde", "Rayo Sur", "Castilla Roja",
            "Granada Luna", "Betis Oeste", "Osasuna Este"],
}
BOOKS = {  # margem abertura, margem fecho, ruído abertura, ruído fecho, viés longshot
    "PS": (0.025, 0.020, 0.050, 0.025, 1.00),
    "B365": (0.055, 0.050, 0.055, 0.035, 0.94),
    "BW": (0.060, 0.055, 0.060, 0.040, 0.93),
    "WH": (0.065, 0.060, 0.060, 0.040, 0.93),
    "IW": (0.065, 0.060, 0.065, 0.045, 0.93),
}
HOME, RHO = 0.27, -0.07
SPORT = {"P1": "soccer_portugal_primeira_liga", "SP1": "soccer_spain_la_liga", "UCL": "soccer_uefa_champs_league"}
API_BOOKS = {"PS": ("pinnacle", "Pinnacle"), "B365": ("betclic", "Betclic"), "BW": ("unibet_eu", "Unibet"),
             "WH": ("williamhill", "William Hill"), "IW": ("sport888", "888sport")}


def _matrix(lam, mu, rho=RHO):
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho; m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho; m[1, 1] *= 1 - rho
    return m / m.sum()


def _probs(m):
    g = np.arange(m.shape[0]); tot = g[:, None] + g[None, :]
    over = m[tot > 2.5].sum()
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]), np.array([over, 1 - over])


def _price(p, rng, margin, noise, flb):
    lp = np.log(p) + rng.normal(0, noise, len(p))
    b = np.exp(lp); b /= b.sum()
    q = b ** flb
    q = q / q.sum() * (1 + margin)
    return [round(float(1 / x), 2) for x in q]


class World:
    def __init__(self, root: Path, seed: int = 7):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.state_p = self.root / "world.json"
        self.rng = np.random.default_rng(seed)
        self.s: dict = {}

    # --------------------------------------------------------------- state
    def load(self):
        self.s = json.loads(self.state_p.read_text())
        self.rng = np.random.default_rng(self.s["seed"] + self.s["round_counter"])
        return self

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_p.write_text(json.dumps(self.s))

    @property
    def now(self) -> datetime:
        return datetime.fromisoformat(self.s["now"])

    # ---------------------------------------------------------------- init
    def init(self, past_seasons: int = 3, played_rounds: int = 8):
        self.raw.mkdir(parents=True, exist_ok=True)
        for f in self.raw.glob("*.csv"):
            f.unlink()
        (self.raw / "DEMO").write_text("dados sintéticos")
        strength = {}
        for lg, teams in TEAMS.items():
            for t in teams:
                strength[t] = [float(self.rng.normal(0.05, 0.28)), float(self.rng.normal(-0.05, 0.22))]
        start_year = 2026 - past_seasons
        self.s = {"seed": 7, "round_counter": 0, "strength": strength, "pending": [],
                  "season_year": start_year, "round": 0, "now": f"{start_year}-08-01T09:00:00"}
        for _ in range(past_seasons):
            self._play_season_rounds(None)
            self._new_season()
        self._play_season_rounds(played_rounds)
        self._make_fixtures()
        self.save()

    def _schedule(self, teams, season_year):
        """Calendário double round-robin (método do círculo)."""
        rng = np.random.default_rng(season_year)
        t = list(teams); rng.shuffle(t)
        n = len(t); rounds = []
        for r in range(n - 1):
            pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
            pairs = [(b, a) if (r + i) % 2 else (a, b) for i, (a, b) in enumerate(pairs)]
            rounds.append(pairs)
            t = [t[0]] + [t[-1]] + t[1:-1]
        return rounds + [[(b, a) for a, b in rp] for rp in rounds]

    def _round_date(self, season_year, rnd):
        d0 = datetime(season_year, 8, 9, 18, 0)
        return d0 + timedelta(days=7 * rnd)

    def _new_season(self):
        self.s["season_year"] += 1
        self.s["round"] = 0
        for t, (a, d) in self.s["strength"].items():  # regressão à média entre épocas
            self.s["strength"][t] = [0.85 * a + float(self.rng.normal(0, 0.07)),
                                     0.85 * d + float(self.rng.normal(0, 0.06))]

    def _csv_path(self, lg):
        y = self.s["season_year"]
        return self.raw / f"{lg}_{y % 100:02d}{(y + 1) % 100:02d}.csv"

    # ----------------------------------------------------------- matches
    def _match_truth(self, h, a, news=None):
        sh, sa = self.s["strength"][h], self.s["strength"][a]
        lam = math.exp(sh[0] + sa[1] + HOME)
        mu = math.exp(sa[0] + sh[1])
        if news:
            if news["team"] == h:
                lam *= news["att_mult"]; mu *= news["def_mult"]
            else:
                mu *= news["att_mult"]; lam *= news["def_mult"]
        return lam, mu

    def _maybe_news(self, h, a):
        if self.rng.random() > 0.18:
            return None
        team = h if self.rng.random() < 0.5 else a
        kind = self.rng.choice(["striker_out", "keeper_out", "rotation", "coach_sacked"])
        mult = {"striker_out": (0.82, 1.0), "keeper_out": (1.0, 1.12),
                "rotation": (0.88, 1.10), "coach_sacked": (1.05, 0.97)}[kind]
        txt = {
            "striker_out": f"Melhor marcador do {team} lesionado, fora do jogo.",
            "keeper_out": f"Guarda-redes titular do {team} castigado; joga o suplente.",
            "rotation": f"{team} vai rodar o onze a pensar na Taça a meio da semana.",
            "coach_sacked": f"{team} despediu o treinador esta semana; interino no banco.",
        }[kind]
        return {"team": team, "kind": str(kind), "att_mult": mult[0], "def_mult": mult[1], "text": txt}

    def _odds_row(self, p1x2, pou, closing: bool):
        row = {}
        for b, (mo, mc, no, nc, flb) in BOOKS.items():
            m, n = (mc, nc) if closing else (mo, no)
            pre = b + ("C" if closing else "")
            row.update(dict(zip([pre + "H", pre + "D", pre + "A"], _price(p1x2, self.rng, m, n, flb))))
        ps = [row[f"PS{'C' if closing else ''}{x}"] for x in "HDA"]
        allb = [[row[f"{b}{'C' if closing else ''}{x}"] for x in "HDA"] for b in BOOKS]
        pre = "C" if closing else ""
        for i, x in enumerate("HDA"):
            row[f"Max{pre}{x}"] = max(o[i] for o in allb)
            row[f"Avg{pre}{x}"] = round(float(np.mean([o[i] for o in allb])), 2)
        ou_ps = _price(pou, self.rng, 0.025 if not closing else 0.02, 0.06 if not closing else 0.03, 1.0)
        ou_b = _price(pou, self.rng, 0.06, 0.09 if not closing else 0.06, 0.95)
        if closing:
            row.update({"PC>2.5": ou_ps[0], "PC<2.5": ou_ps[1], "B365C>2.5": ou_b[0], "B365C<2.5": ou_b[1],
                        "MaxC>2.5": max(ou_ps[0], ou_b[0]), "MaxC<2.5": max(ou_ps[1], ou_b[1]),
                        "AvgC>2.5": round((ou_ps[0] + ou_b[0]) / 2, 2), "AvgC<2.5": round((ou_ps[1] + ou_b[1]) / 2, 2)})
        else:
            row.update({"P>2.5": ou_ps[0], "P<2.5": ou_ps[1], "B365>2.5": ou_b[0], "B365<2.5": ou_b[1],
                        "Max>2.5": max(ou_ps[0], ou_b[0]), "Max<2.5": max(ou_ps[1], ou_b[1]),
                        "Avg>2.5": round((ou_ps[0] + ou_b[0]) / 2, 2), "Avg<2.5": round((ou_ps[1] + ou_b[1]) / 2, 2)})
        _ = ps
        return row

    def _fixture_list(self):
        out = []
        for lg, teams in TEAMS.items():
            sched = self._schedule(teams, self.s["season_year"])
            if self.s["round"] >= len(sched):
                return []
            d = self._round_date(self.s["season_year"], self.s["round"])
            for k, (h, a) in enumerate(sched[self.s["round"]]):
                out.append({"div": lg, "date": (d + timedelta(hours=(k % 4) * 2 - 3 + 24 * (k % 2))).isoformat(),
                            "home": h, "away": a})
        return out

    def _make_fixtures(self):
        """Cria a próxima jornada com odds de abertura e notícias escondidas."""
        fx = self._fixture_list()
        rows, pending = [], []
        for f in fx:
            lam, mu = self._match_truth(f["home"], f["away"])
            p1, pou = _probs(_matrix(lam, mu))
            news = self._maybe_news(f["home"], f["away"])
            d = datetime.fromisoformat(f["date"])
            r = {"Div": f["div"], "Date": d.strftime("%d/%m/%Y"), "Time": d.strftime("%H:%M"),
                 "HomeTeam": f["home"], "AwayTeam": f["away"]}
            r.update(self._odds_row(p1, pou, closing=False))
            rows.append(r)
            pending.append({**f, "news": news, "open": r})
        pd.DataFrame(rows).to_csv(self.raw / "fixtures.csv", index=False)
        if fx:
            first = min(datetime.fromisoformat(f["date"]) for f in fx)
            # jornada europeia: 6 jogos entre as duas ligas, na véspera
            teams_p, teams_s = list(TEAMS["P1"]), list(TEAMS["SP1"])
            self.rng.shuffle(teams_p); self.rng.shuffle(teams_s)
            for k in range(6):
                h, a = (teams_p[k], teams_s[k]) if k % 2 == 0 else (teams_s[k], teams_p[k])
                d = first - timedelta(hours=22 - 2 * (k % 3))
                f = {"div": "UCL", "date": d.isoformat(), "home": h, "away": a}
                lam, mu = self._match_truth(h, a)
                p1, pou = _probs(_matrix(lam, mu))
                pending.append({**f, "news": self._maybe_news(h, a), "open": self._odds_row(p1, pou, closing=False)})
            self.s["now"] = (first - timedelta(hours=30)).isoformat()
        self.s["pending"] = pending
        self._write_api_files()
        self._write_elo()

    def news_for(self, div, home, away):
        for p in self.s.get("pending", []):
            if p["home"] == home and p["away"] == away:
                return p.get("news")
        return None

    # ------------------------------------------- formato Odds API / ClubElo
    @property
    def api_dir(self):
        return self.raw / "oddsapi"

    def _write_api_files(self):
        d = self.api_dir
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("odds_*.json"):
            f.unlink()
        for f in d.glob("event_*.json"):
            f.unlink()
        by = {}
        for p in self.s["pending"]:
            o = p["open"]
            if "id" not in p:
                p["id"] = uuid.uuid4().hex
            t = datetime.fromisoformat(p["date"]).replace(tzinfo=ZoneInfo("Europe/Lisbon")).astimezone(timezone.utc)
            books = []
            for code, (key, title) in API_BOOKS.items():
                h2h = [o[f"{code}H"], o[f"{code}D"], o[f"{code}A"]]
                ou = [o.get(f"{'P' if code == 'PS' else code}>2.5"), o.get(f"{'P' if code == 'PS' else code}<2.5")]
                mk = [{"key": "h2h", "outcomes": [{"name": p["home"], "price": h2h[0]}, {"name": "Draw", "price": h2h[1]},
                                                  {"name": p["away"], "price": h2h[2]}]}]
                if ou[0] and ou[1]:
                    mk.append({"key": "totals", "outcomes": [{"name": "Over", "price": ou[0], "point": 2.5},
                                                             {"name": "Under", "price": ou[1], "point": 2.5}]})
                books.append({"key": key, "title": title, "markets": mk})
            ev = {"id": p["id"], "sport_key": SPORT[p["div"]], "commence_time": t.isoformat().replace("+00:00", "Z"),
                  "home_team": p["home"], "away_team": p["away"], "bookmakers": books}
            by.setdefault(SPORT[p["div"]], []).append(ev)
            # mercados extra (ambas marcam, handicap asiático) a partir da verdade sem notícias
            lam, mu = self._match_truth(p["home"], p["away"])
            m = _matrix(lam, mu)
            pb = float(m[1:, 1:].sum())
            gd = {}
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    gd[i - j] = gd.get(i - j, 0) + m[i, j]
            def ah(line):
                w = sum(v for k, v in gd.items() if k + line > 0); l = sum(v for k, v in gd.items() if k + line < 0)
                return w / (w + l)
            main = min([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], key=lambda L: abs(ah(L) - 0.5))
            extra = []
            for code in ("PS", "B365"):
                key, title = API_BOOKS[code]
                mg, nz = (0.025, 0.03) if code == "PS" else (0.06, 0.05)
                bt = _price(np.array([pb, 1 - pb]), self.rng, mg, nz, 1.0)
                ahp = _price(np.array([ah(main), 1 - ah(main)]), self.rng, mg, nz, 1.0)
                extra.append({"key": key, "title": title, "markets": [
                    {"key": "btts", "outcomes": [{"name": "Yes", "price": bt[0]}, {"name": "No", "price": bt[1]}]},
                    {"key": "spreads", "outcomes": [{"name": p["home"], "price": ahp[0], "point": main},
                                                    {"name": p["away"], "price": ahp[1], "point": -main}]}]})
            (d / f"event_{p['id']}.json").write_text(json.dumps({"id": p["id"], "bookmakers": extra}))
        for sk, evs in by.items():
            (d / f"odds_{sk}.json").write_text(json.dumps(evs))

    def _write_elo(self):
        rows = []
        for lg, teams in TEAMS.items():
            for t in teams:
                a, dfn = self.s["strength"][t]
                rows.append({"Rank": 0, "Club": t, "Country": "POR" if lg == "P1" else "ESP", "Level": 1,
                             "Elo": round(1550 + 520 * (a - dfn) + float(self.rng.normal(0, 25)), 1)})
        pd.DataFrame(rows).to_csv(self.raw / "clubelo.csv", index=False)
        # "Understat" sintético: xG dos jogos já disputados, no formato do getLeagueData
        for lg in TEAMS:
            dates = []
            for f in sorted(self.raw.glob(f"{lg}_*.csv")):
                d = pd.read_csv(f)
                if "XGH" not in d:
                    continue
                for r in d.itertuples():
                    dt = datetime.strptime(f"{r.Date} {r.Time}", "%d/%m/%Y %H:%M")
                    dates.append({"isResult": True, "h": {"title": r.HomeTeam}, "a": {"title": r.AwayTeam},
                                  "goals": {"h": str(r.FTHG), "a": str(r.FTAG)}, "xG": {"h": str(r.XGH), "a": str(r.XGA)},
                                  "datetime": dt.strftime("%Y-%m-%d %H:%M:%S")})
            (self.raw / f"understat_{lg}.json").write_text(json.dumps({"dates": dates}))

    def _play(self, f, open_row=None, news=None):
        lam, mu = self._match_truth(f["home"], f["away"], news)
        m = _matrix(lam, mu)
        p1, pou = _probs(m)
        k = self.rng.choice(m.size, p=m.ravel())
        hg, ag = divmod(int(k), m.shape[1])
        d = datetime.fromisoformat(f["date"])
        row = {"Div": f["div"], "Date": d.strftime("%d/%m/%Y"), "Time": d.strftime("%H:%M"),
               "HomeTeam": f["home"], "AwayTeam": f["away"], "FTHG": hg, "FTAG": ag,
               "FTR": "H" if hg > ag else ("D" if hg == ag else "A"),
               "HS": hg + int(self.rng.poisson(6 * lam + 3)), "AS": ag + int(self.rng.poisson(6 * mu + 3)),
               "HST": hg + int(self.rng.poisson(2.4 * lam + 0.6)), "AST": ag + int(self.rng.poisson(2.4 * mu + 0.6)),
               "XGH": round(float(lam * self.rng.gamma(8, 1 / 8)), 2), "XGA": round(float(mu * self.rng.gamma(8, 1 / 8)), 2)}
        if open_row is None:
            # jogos históricos: abertura sem notícias, gerada agora
            lam0, mu0 = self._match_truth(f["home"], f["away"])
            p10, pou0 = _probs(_matrix(lam0, mu0))
            open_row = self._odds_row(p10, pou0, closing=False)
        row.update({k2: v for k2, v in open_row.items() if k2 not in row})
        # fecho: o mercado incorpora ~70% da notícia
        if news:
            lam_c, mu_c = self._match_truth(f["home"], f["away"])
            lam_n, mu_n = lam, mu
            lam_c = lam_c * (lam_n / lam_c) ** 0.7; mu_c = mu_c * (mu_n / mu_c) ** 0.7
            pc1, pcou = _probs(_matrix(lam_c, mu_c))
        else:
            pc1, pcou = p1, pou
        row.update(self._odds_row(pc1, pcou, closing=True))
        return row

    def _append_results(self, rows):
        by = {}
        for r in rows:
            by.setdefault(r["Div"], []).append(r)
        for lg, rs in by.items():
            p = self._csv_path(lg)
            new = pd.DataFrame(rs)
            if p.exists():
                new = pd.concat([pd.read_csv(p), new], ignore_index=True)
            new.to_csv(p, index=False)

    def _drift(self):
        for t, (a, d) in self.s["strength"].items():
            self.s["strength"][t] = [a + float(self.rng.normal(0, 0.012)), d + float(self.rng.normal(0, 0.010))]

    def _play_season_rounds(self, n_rounds):
        played = 0
        while n_rounds is None or played < n_rounds:
            fx = self._fixture_list()
            if not fx:
                break
            rows = []
            for f in fx:
                news = self._maybe_news(f["home"], f["away"])
                rows.append(self._play(f, None, news))
            self._append_results(rows)
            self._drift()
            self.s["round"] += 1
            played += 1

    def advance(self):
        """Joga a jornada pendente (com as notícias escondidas) e cria a seguinte."""
        self.s["round_counter"] += 1
        rows, scores = [], []
        for p in self.s.get("pending", []):
            r = self._play(p, p["open"], p.get("news"))
            if p["div"] == "UCL":
                scores.append({"id": p.get("id"), "sport_key": SPORT["UCL"], "completed": True,
                               "commence_time": datetime.fromisoformat(p["date"]).replace(tzinfo=ZoneInfo("Europe/Lisbon")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "home_team": p["home"], "away_team": p["away"],
                               "scores": [{"name": p["home"], "score": str(r["FTHG"])},
                                          {"name": p["away"], "score": str(r["FTAG"])}]})
            else:
                rows.append(r)
        if rows:
            self._append_results(rows)
        self.api_dir.mkdir(parents=True, exist_ok=True)
        prev = []
        sp = self.api_dir / f"scores_{SPORT['UCL']}.json"
        if sp.exists():
            prev = json.loads(sp.read_text())[-60:]
        sp.write_text(json.dumps(prev + scores))
        self._drift()
        self.s["round"] += 1
        if not self._fixture_list():
            self._new_season()
        self._make_fixtures()
        self.save()
        return len(rows)
