"""Nomes de equipas entre fontes diferentes.

O football-data escreve "Sp Lisbon", a Odds API "Sporting Lisbon" e o ClubElo
"Sporting". Tudo passa por uma chave canónica: sem acentos, sem prefixos
como FC/SC/CF e com aliases para os casos conhecidos. Quando não há alias,
usa-se fuzzy matching com um limiar conservador; mais vale "equipa
desconhecida" do que trocar uma equipa por outra.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

DROP = {"fc", "cf", "sc", "ac", "as", "ss", "ssc", "afc", "cd", "ud", "sd", "rc", "rcd", "sv", "vfb", "vfl",
        "tsg", "fk", "sk", "bk", "club", "de", "the", "calcio", "1", "04", "05", "09", "1899", "1846", "1907",
        "sl", "fsv", "us", "ogc", "losc", "stade", "olympique", "sporting", "real", "cp", "sad", "bc", "kv",
        "krc", "rsc", "royal", "if", "aik", "ca", "clube", "futebol", "football", "calcio"}

# variante (já em chave simples) -> canónico
ALIASES = {
    # Portugal
    "sp lisbon": "sporting", "sporting lisbon": "sporting", "sporting cp": "sporting", "sporting clube de portugal": "sporting",
    "sp braga": "braga", "sporting braga": "braga", "sc braga": "braga",
    "guimaraes": "vitoria guimaraes", "vitoria sc": "vitoria guimaraes", "v guimaraes": "vitoria guimaraes",
    "fc porto": "porto", "sl benfica": "benfica", "famalicao": "famalicao", "estoril": "estoril", "estoril praia": "estoril",
    "gil vicente": "gil vicente", "casa pia": "casa pia", "rio ave": "rio ave", "santa clara": "santa clara",
    "moreirense": "moreirense", "arouca": "arouca", "nacional": "nacional", "estrela": "estrela amadora",
    "estrela da amadora": "estrela amadora", "avs": "avs", "afs": "avs", "tondela": "tondela", "alverca": "alverca",
    # Inglaterra
    "man united": "manchester united", "man utd": "manchester united", "manchester utd": "manchester united",
    "man city": "manchester city", "nott m forest": "nottingham forest", "nottm forest": "nottingham forest",
    "wolves": "wolverhampton", "wolverhampton wanderers": "wolverhampton", "tottenham hotspur": "tottenham",
    "spurs": "tottenham", "newcastle united": "newcastle", "west ham united": "west ham",
    "brighton and hove albion": "brighton", "brighton hove albion": "brighton", "leicester city": "leicester",
    "leeds united": "leeds", "sheffield united": "sheffield utd", "sheffield weds": "sheffield wednesday",
    "afc bournemouth": "bournemouth", "ipswich town": "ipswich", "luton town": "luton",
    # Espanha
    "ath madrid": "atletico madrid", "atletico de madrid": "atletico madrid", "atletico": "atletico madrid",
    "ath bilbao": "athletic bilbao", "athletic club": "athletic bilbao", "betis": "real betis",
    "sociedad": "real sociedad", "celta": "celta vigo", "rc celta": "celta vigo", "espanol": "espanyol",
    "vallecano": "rayo vallecano", "rayo": "rayo vallecano", "alaves": "alaves", "deportivo alaves": "alaves",
    "la coruna": "deportivo la coruna", "sevilla fc": "sevilla", "barcelona": "barcelona", "barca": "barcelona",
    "real madrid": "real madrid", "real oviedo": "oviedo", "oviedo": "oviedo",
    # Itália
    "inter": "internazionale", "inter milan": "internazionale", "internazionale milano": "internazionale",
    "milan": "ac milan", "ac milan": "ac milan", "roma": "as roma", "as roma": "as roma", "verona": "hellas verona",
    "hellas verona": "hellas verona", "napoli": "napoli", "ssc napoli": "napoli", "lazio": "lazio",
    # Alemanha
    "m gladbach": "monchengladbach", "borussia monchengladbach": "monchengladbach", "gladbach": "monchengladbach",
    "dortmund": "dortmund", "borussia dortmund": "dortmund", "ein frankfurt": "frankfurt",
    "eintracht frankfurt": "frankfurt", "leverkusen": "leverkusen", "bayer leverkusen": "leverkusen",
    "bayern munich": "bayern", "bayern munchen": "bayern", "fc bayern munchen": "bayern",
    "fc koln": "koln", "1 fc koln": "koln", "cologne": "koln", "mainz": "mainz", "mainz 05": "mainz",
    "hoffenheim": "hoffenheim", "tsg hoffenheim": "hoffenheim", "rb leipzig": "leipzig", "leipzig": "leipzig",
    "union berlin": "union berlin", "fc union berlin": "union berlin", "werder bremen": "bremen", "bremen": "bremen",
    "stuttgart": "stuttgart", "vfb stuttgart": "stuttgart", "wolfsburg": "wolfsburg", "freiburg": "freiburg",
    "sc freiburg": "freiburg", "augsburg": "augsburg", "heidenheim": "heidenheim", "st pauli": "st pauli",
    "hamburg": "hamburg", "hamburger sv": "hamburg",
    # França
    "paris sg": "psg", "paris saint germain": "psg", "paris saint-germain": "psg", "psg": "psg",
    "st etienne": "saint etienne", "marseille": "marseille", "olympique marseille": "marseille",
    "lyon": "lyon", "olympique lyonnais": "lyon", "monaco": "monaco", "as monaco": "monaco", "lille": "lille",
    "rennes": "rennes", "stade rennais": "rennes", "nice": "nice", "lens": "lens", "rc lens": "lens",
    "brest": "brest", "stade brestois": "brest", "strasbourg": "strasbourg",
    # Países Baixos
    "psv eindhoven": "psv", "psv": "psv", "ajax": "ajax", "ajax amsterdam": "ajax", "feyenoord": "feyenoord",
    "az alkmaar": "az", "az": "az", "twente": "twente", "fc twente": "twente", "utrecht": "utrecht",
    # Outros
    "club brugge": "brugge", "club brugge kv": "brugge", "brugge": "brugge", "salzburg": "salzburg",
    "rb salzburg": "salzburg", "red bull salzburg": "salzburg", "celtic": "celtic", "rangers": "rangers",
    "galatasaray": "galatasaray", "fenerbahce": "fenerbahce", "besiktas": "besiktas", "olympiakos": "olympiakos",
    "olympiacos": "olympiakos", "shakhtar donetsk": "shakhtar", "dinamo zagreb": "dinamo zagreb",
    "crvena zvezda": "crvena zvezda", "red star belgrade": "crvena zvezda", "slavia praha": "slavia prague",
    "slavia prague": "slavia prague", "sparta praha": "sparta prague", "sparta prague": "sparta prague",
    "young boys": "young boys", "bsc young boys": "young boys", "copenhagen": "kobenhavn", "fc copenhagen": "kobenhavn",
}


def simple(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def key(name: str) -> str:
    s = simple(name)
    if s in ALIASES:
        return ALIASES[s]
    toks = [t for t in s.split() if t not in DROP]
    k = " ".join(toks) or s
    return ALIASES.get(k, k)


def match(name: str, candidates, cutoff: float = 0.82) -> str | None:
    """Encontra `name` numa lista de nomes de outra fonte. Devolve o nome original ou None."""
    cands = list(candidates)
    if not cands:
        return None
    k = key(name)
    by_key = {}
    for c in cands:
        by_key.setdefault(key(c), c)
    if k in by_key:
        return by_key[k]
    # contido (ex.: "olympique lyonnais" vs "lyon")
    for ck, c in by_key.items():
        if len(k) >= 4 and len(ck) >= 4 and (k in ck.split() or ck in k.split()):
            return c
    best = difflib.get_close_matches(k, list(by_key), n=1, cutoff=cutoff)
    return by_key[best[0]] if best else None


def same(a: str, b: str) -> bool:
    return key(a) == key(b) or match(a, [b]) is not None
