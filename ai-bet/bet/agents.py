"""Os agentes LLM: prompts, esquemas e as versões "mock" usadas no demo.

Papel dos LLMs, com base no que a literatura mostra:
  - Sozinhos não batem o mercado. Nos benchmarks do Mundial 2026 nenhum LLM
    bateu o Brier do mercado e apostar contra o favorito deu prejuízo a todos.
  - Valem como RECOLHA de informação recente (Halawi et al. 2024: retrieval é
    o que torna os LLMs bons a prever) e como agregação de vários palpites
    (Schoenegger et al. 2024: "wisdom of the silicon crowd").
Por isso aqui os LLMs NÃO dão probabilidades a olho. Recolhem factos com
fontes e traduzem-nos em multiplicadores de xG LIMITADOS, aplicados sobre o
modelo quantitativo. No fim, o boletim de notas mede se ajudaram ou não.
"""
from __future__ import annotations

import json
import random
import statistics

from . import llm

LIM = (0.75, 1.25)


def _clip(x, lo=LIM[0], hi=LIM[1]):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 1.0


def _match_header(m: dict) -> str:
    return (f"Jogo: {m['home']} (casa) vs {m['away']} (fora) — {m.get('comp', m.get('div'))}, "
            f"{m['date'][:16].replace('T', ' ')} (hora de Lisboa aprox.).")


# --------------------------------------------------------------- MÉDICO
def medico(m: dict, cfg: dict, mock_news=None):
    if cfg["llm"]["backend"] == "mock":
        return _mock_medico(m, mock_news)
    p = f"""{_match_header(m)}
Tarefa (MÉDICO): pesquisa na web notícias dos últimos 7 dias sobre LESÕES, CASTIGOS e
ONZE PROVÁVEL das duas equipas. Usa fontes desportivas fiáveis (sites dos clubes,
jornais desportivos, conferências de imprensa). Para cada ausência indica a importância
(3 = titular indiscutível/melhor jogador, 2 = titular, 1 = rotação) e a DATA da notícia.

Esquema:
```json
{{"home": {{"absences": [{{"player": "", "position": "GR|DEF|MED|AV", "importance": 1,
            "status": "fora|dúvida", "news_date": "AAAA-MM-DD", "source": "url"}}],
           "lineup_notes": ""}},
  "away": {{"absences": [], "lineup_notes": ""}},
  "searched": ["consultas que fizeste"], "confidence": 0.0}}
```"""
    return llm.ask(p, cfg["llm"]["backend"], cfg["llm"]["scout_model"], web=True,
                   timeout=cfg["llm"].get("scout_timeout", 420))


def _mock_medico(m, news):
    out = {"home": {"absences": [], "lineup_notes": "sem novidades"},
           "away": {"absences": [], "lineup_notes": "sem novidades"},
           "searched": ["(demo) notícias sintéticas"], "confidence": 0.7}
    if news and news["kind"] in ("striker_out", "keeper_out"):
        side = "home" if news["team"] == m["home"] else "away"
        out[side]["absences"].append({
            "player": "Melhor marcador" if news["kind"] == "striker_out" else "GR titular",
            "position": "AV" if news["kind"] == "striker_out" else "GR", "importance": 3,
            "status": "fora", "news_date": m["date"][:10], "source": "demo://noticia"})
        out[side]["lineup_notes"] = news["text"]
    return out


# ------------------------------------------------------------- REPÓRTER
def reporter(m: dict, cfg: dict, mock_news=None):
    if cfg["llm"]["backend"] == "mock":
        return _mock_reporter(m, mock_news)
    p = f"""{_match_header(m)}
Tarefa (REPÓRTER): pesquisa na web o CONTEXTO deste jogo, com notícias dos últimos 7 dias:
motivação (luta pelo título, Europa, descida, nada em jogo), calendário (jogos europeus ou
de taça nos 4 dias antes ou depois, risco de rotação), mudanças de treinador, crises
internas, viagens longas e a previsão do tempo para o estádio (chuva forte, vento).
Não repitas lesões; isso é trabalho do médico. Dá a DATA de cada notícia.

Esquema:
```json
{{"home": {{"motivation": "", "rotation_risk": 0, "coach_change": false, "notes": []}},
  "away": {{"motivation": "", "rotation_risk": 0, "coach_change": false, "notes": []}},
  "weather": "", "other": [], "sources": ["url"], "confidence": 0.0}}
```
rotation_risk: 0 = nenhum, 3 = vai jogar a equipa B."""
    return llm.ask(p, cfg["llm"]["backend"], cfg["llm"]["scout_model"], web=True,
                   timeout=cfg["llm"].get("scout_timeout", 420))


def _mock_reporter(m, news):
    out = {"home": {"motivation": "normal", "rotation_risk": 0, "coach_change": False, "notes": []},
           "away": {"motivation": "normal", "rotation_risk": 0, "coach_change": False, "notes": []},
           "weather": random.choice(["seco, 18°C", "aguaceiros fracos", "céu limpo", "vento moderado"]),
           "other": [], "sources": ["demo://contexto"], "confidence": 0.6}
    if news and news["kind"] in ("rotation", "coach_sacked"):
        side = "home" if news["team"] == m["home"] else "away"
        if news["kind"] == "rotation":
            out[side]["rotation_risk"] = 2
        else:
            out[side]["coach_change"] = True
        out[side]["notes"].append(news["text"])
    return out


# ---------------------------------------------------------------- RADAR
def radar(league_name: str, fixtures: list[dict], cfg: dict, world=None):
    """Uma pesquisa por liga e por ronda que varre TODOS os jogos à procura de
    notícias fortes. Os jogos assinalados passam a triagem mesmo sem edge
    estatística, porque é aí que o preço de abertura pode estar desatualizado."""
    if cfg["llm"]["backend"] == "mock":
        out = []
        for i, f in enumerate(fixtures):
            n = world.news_for(f["div"], f["home"], f["away"]) if world else None
            if n:
                out.append({"i": i, "headline": n["text"], "impact": 2})
        return {"flags": out}
    lst = "\n".join(f"{i}: {f['home']} vs {f['away']} ({f['date'][:10]})" for i, f in enumerate(fixtures))
    p = f"""Liga: {league_name}. Jogos desta ronda:
{lst}

Tarefa (RADAR): faz UMA varredura rápida na web por notícias das últimas 72h com impacto forte
nestes jogos: lesão ou castigo de um jogador-chave, guarda-redes titular fora, treinador
despedido, rotação anunciada por causa de jogos europeus, crise grave. Ignora rumores fracos.
impact: 1 = pequeno, 2 = relevante, 3 = muito forte.
```json
{{"flags": [{{"i": 0, "headline": "", "impact": 2, "source": "url"}}]}}
```"""
    return llm.ask(p, cfg["llm"]["backend"], cfg["llm"]["scout_model"], web=True,
                   timeout=cfg["llm"].get("scout_timeout", 420))


# ------------------------------------------------------------ AJUSTADOR
def ajustador(m: dict, scouts: dict, cfg: dict, seed=0):
    """Traduz os factos em multiplicadores de xG. É corrido N vezes e fica a mediana."""
    if cfg["llm"]["backend"] == "mock":
        return _mock_ajustador(m, scouts, seed)
    c = m["card"]
    q = {"xg_casa": round(c["lam"], 2), "xg_fora": round(c["mu"], 2), "fontes": c["components"],
         "forma": c.get("form"), "descanso_dias": c.get("rest_days"), "elo": c.get("elo"),
         "prob_mercado_1x2": c.get("p1x2_mkt")}
    p = f"""{_match_header(m)}
Os modelos estimam xG casa={c['lam']:.2f}, fora={c['mu']:.2f}. Estes números já incluem as odds
do mercado, a forma e a força das equipas, mas NÃO sabem das notícias de última hora.
Dados: {json.dumps(q, ensure_ascii=False, default=str)}
Relatórios dos batedores: {json.dumps(scouts, ensure_ascii=False)}

Tarefa (ANALISTA DE NOTÍCIAS): converte só os factos com fonte em multiplicadores de xG.
Regras:
- Referências: avançado titular/melhor marcador fora → ataque ~0.85-0.92; GR titular fora →
  xG sofrido ~1.05-1.10; rotação forte → ataque ~0.88 e defesa ~1.08. Sem novidades → 1.0.
- Limites: entre 0.75 e 1.25. Na dúvida, aproxima de 1.0. Não "adivinhes" qualidade.
- priced_in (0-1): probabilidade de o mercado já refletir a notícia. Notícias com mais de 2
  dias ou muito mediáticas estão quase sempre no preço.
Esquema:
```json
{{"home_attack": 1.0, "home_defense": 1.0, "away_attack": 1.0, "away_defense": 1.0,
  "priced_in": 0.5, "confidence": 0.5, "reasons": ["facto → efeito"]}}
```
(home_defense > 1 = a casa vai sofrer mais golos)"""
    return llm.ask(p, cfg["llm"]["backend"], cfg["llm"]["model"], timeout=240)


def _mock_ajustador(m, scouts, seed):
    rng = random.Random(hash((m["mid"], seed)))
    out = {"home_attack": 1.0, "home_defense": 1.0, "away_attack": 1.0, "away_defense": 1.0,
           "priced_in": 0.3, "confidence": 0.6, "reasons": []}
    for side in ("home", "away"):
        for a in scouts.get("medico", {}).get(side, {}).get("absences", []):
            if a["position"] == "AV":
                out[f"{side}_attack"] *= rng.uniform(0.82, 0.92)
                out["reasons"].append(f"{a['player']} ({side}) fora → menos golos marcados")
            if a["position"] == "GR":
                out[f"{side}_defense"] *= rng.uniform(1.05, 1.13)
                out["reasons"].append(f"GR titular ({side}) fora → mais golos sofridos")
        rep = scouts.get("reporter", {}).get(side, {})
        if rep.get("rotation_risk", 0) >= 2:
            out[f"{side}_attack"] *= rng.uniform(0.85, 0.93)
            out[f"{side}_defense"] *= rng.uniform(1.04, 1.12)
            out["reasons"].append(f"rotação provável ({side})")
        if rep.get("coach_change"):
            out[f"{side}_attack"] *= rng.uniform(1.0, 1.06)
            out["reasons"].append(f"chicotada psicológica ({side})")
    if not out["reasons"]:
        out["reasons"] = ["sem notícias relevantes → sem ajuste"]
        out["confidence"] = 0.9
    return out


def aggregate_adjust(samples: list[dict]) -> dict:
    """Mediana das amostras (sabedoria da multidão) + limites."""
    samples = [s for s in samples if s]
    if not samples:
        return {"lam": 1.0, "mu": 1.0, "priced_in": 1.0, "confidence": 0.0, "reasons": [], "n": 0}
    med = lambda k, d=1.0: statistics.median([_clip(s.get(k, d), 0, 2) for s in samples])
    ha, hd, aa, ad = med("home_attack"), med("home_defense"), med("away_attack"), med("away_defense")
    lam = _clip(ha * ad)
    mu = _clip(aa * hd)
    pin = statistics.median([min(1, max(0, float(s.get("priced_in", 0.5)))) for s in samples])
    conf = statistics.median([min(1, max(0, float(s.get("confidence", 0.5)))) for s in samples])
    reasons = []
    for s in samples:
        for r in s.get("reasons", []):
            if r not in reasons:
                reasons.append(r)
    spread = max(abs(_clip(s.get("home_attack", 1)) - ha) for s in samples)
    return {"lam": lam, "mu": mu, "priced_in": pin, "confidence": conf, "reasons": reasons[:6],
            "n": len(samples), "disagreement": round(spread, 3)}


# --------------------------------------------------------------- CÉTICO
def _fmt_c(i, c):
    return {"id": i, "aposta": c["label"], "odd": c["odds"], "odd_estimada": c.get("estimated", False),
            "casa": c.get("book"), "prob_nossa": round(c["p"], 3), "prob_mercado": round(c["p_mkt"], 3),
            "prob_so_modelos": None if c.get("p_models") is None else round(c["p_models"], 3),
            "edge": round(c["edge"], 3), "odd_justa": round(c["fair"], 2)}


def cetico(m: dict, cands: list[dict], cfg: dict):
    if cfg["llm"]["backend"] == "mock":
        return _mock_cetico(m, cands)
    c = m["card"]
    info = {"xg": [round(c["lam"], 2), round(c["mu"], 2)], "fontes_xg": c["components"],
            "ajuste_noticias": m.get("adjust"), "forma": c.get("form"), "descanso": c.get("rest_days"),
            "elo": c.get("elo"), "radar": m.get("radar")}
    p = f"""{_match_header(m)}
O sistema está a pensar recomendar estas apostas: {json.dumps([_fmt_c(i, x) for i, x in enumerate(cands)], ensure_ascii=False)}
Informação: {json.dumps(info, ensure_ascii=False, default=str)}

Tarefa (ADVOGADO DO DIABO): o mercado costuma ter razão. Para cada aposta, procura a explicação
mais provável para a "edge" ser ilusória: notícia já no preço, modelo cego a algo (equipa
promovida, treinador novo, jogo sem importância, rotação), odd de uma só casa fora do consenso,
odd estimada demasiado otimista, etc.
severity: 0 = credível, 1 = dúvida leve, 2 = fraca, 3 = não recomendar.
```json
{{"reviews": [{{"id": 0, "severity": 0, "argument": ""}}]}}
```"""
    return llm.ask(p, cfg["llm"]["backend"], cfg["llm"]["model"], timeout=240)


def _mock_cetico(m, cands):
    rev = []
    adj = m.get("adjust") or {}
    for i, c in enumerate(cands):
        sev, arg = 0, "Edge coerente com os modelos e sem sinais de alarme."
        if c["edge"] > 0.12 and not m.get("has_news"):
            sev, arg = 2, "Edge grande sem notícia que a explique: pode ser erro do modelo."
        elif c.get("estimated"):
            sev, arg = 1, "A odd é estimada: confirma na tua casa que chega à odd mínima."
        elif c.get("p_models") is not None and (c["p_models"] - c["p_mkt"]) * (c["p"] - c["p_mkt"]) < 0:
            sev, arg = 1, "Os modelos estatísticos puxam para o lado contrário; o valor vem só de uma odd fora do consenso."
        rev.append({"id": i, "severity": sev, "argument": arg})
    return {"reviews": rev}


# ----------------------------------------------------------- PRESIDENTE
def presidente(cands: list[dict], proposal: dict, cfg: dict):
    """Fecha o boletim a partir das candidatas e escreve o porquê de cada uma."""
    if cfg["llm"]["backend"] == "mock":
        return _mock_presidente(cands, proposal)
    cs = []
    for i, c in enumerate(cands):
        d = _fmt_c(i, c)
        d.update({"jogo": f"{c['home']} vs {c['away']}", "competicao": c["comp"], "hora": c["date"][:16],
                  "tipo": c.get("kind"), "estrelas": c.get("stars"), "cetico": (c.get("skeptic") or {}).get("argument"),
                  "noticias": c.get("news_reasons"), "forma": c.get("form_txt"), "classificacao": c.get("table"),
                  "xg_ultimos_jogos": c.get("xg_form"), "confrontos_diretos": c.get("h2h"),
                  "sem_verificacao_de_noticias": bool(c.get("unchecked"))})
        cs.append(d)
    n = cfg.get("boletim", {}).get("singles", 5)
    p = f"""Candidatas de hoje (tipo "valor" = odd real acima da justa; "solida" = provável e o sistema
concorda ou está acima do mercado; "segura" = muito provável, odd baixa):
{json.dumps(cs, ensure_ascii=False, default=str)}
Proposta do sistema: simples={proposal['singles']}, mais_provaveis={proposal.get('safe')}, multipla={proposal.get('multi')}

Tarefa (PRESIDENTE): fecha o BOLETIM DO DIA de um amigo que aposta por gosto.
- "singles": até {n} ids, no máximo 1 por jogo, com prioridade às de tipo "valor" e às sólidas
  com mais apoio (modelos, xG, notícias). Se houver candidatas, NÃO deixes a lista vazia.
- "multi": 2 a 4 ids de jogos diferentes, prováveis (podes usar as "segura"), ou mantém a proposta.
- "why": para CADA id das simples, das mais prováveis e da múltipla, 1-2 frases em português de
  Portugal, descontraídas e concretas (números, forma, xG, classificação, notícias). Não prometas nada.
- "intro": introdução curta ao dia.
```json
{{"singles": [0], "multi": [0, 1], "why": {{"0": ""}}, "intro": ""}}
```"""
    return llm.ask(p, cfg["llm"]["backend"], cfg["llm"]["pm_model"], timeout=300)


def _mock_presidente(cands, proposal):
    why = {}
    for i in set(proposal["singles"]) | set(proposal.get("multi") or []) | set(proposal.get("safe") or []):
        c = cands[i]
        bits = []
        if c.get("news_reasons"):
            bits.append(c["news_reasons"][0].capitalize())
        xf = c.get("xg_form") or {}
        if xf.get("home") and xf.get("away"):
            bits.append(f"xG recente {xf['home']['xg_for']:.1f} vs {xf['away']['xg_for']:.1f} por jogo")
        if c.get("p_models") is not None and c["p_models"] - c["p_mkt"] > 0.02:
            bits.append(f"os modelos dão {c['p_models']:.0%} contra {c['p_mkt']:.0%} do mercado")
        if c.get("form_txt"):
            bits.append(c["form_txt"])
        bits.append(f"{c['p']:.0%} de probabilidade")
        t = "; ".join(bits[:3])
        why[str(i)] = t[:1].upper() + t[1:] + "."
    n = len(proposal["singles"])
    intro = (f"{n} apostas para os próximos dias, as mais prováveis da jornada e uma múltipla de favoritos. "
             "Confirma sempre a odd mínima na tua casa.") if n else "Poucos jogos com interesse nos próximos dias."
    return {"singles": proposal["singles"], "multi": proposal.get("multi") or [], "why": why, "intro": intro}
