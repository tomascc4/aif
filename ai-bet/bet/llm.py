"""Chamadas aos LLMs pelos CLIs, que usam as tuas subscrições e não precisam de API key.

  claude: `claude -p` (Claude Code CLI). Os batedores recebem as ferramentas
          WebSearch/WebFetch para ir buscar notícias de hoje.
  codex:  `codex exec` (subscrição ChatGPT), sem pesquisa web.
  mock:   devolve None e os agentes usam a versão de demonstração.

Os agentes devolvem sempre JSON. Nenhum agente aposta: as apostas são feitas
por código (risk.py), dentro dos limites.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

RULES = (
    "És parte de um sistema de investigação multi-agente de PAPER betting em futebol "
    "(sem dinheiro real). Não faças perguntas. Responde APENAS com um objeto JSON dentro "
    "de um bloco ```json, a seguir o esquema pedido. Sê factual: se não encontrares algo, diz "
    "que não encontraste em vez de inventar."
)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else None
    if raw is None:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise LLMError("sem JSON na resposta")
        raw = text[a:b + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON inválido: {e}") from e


def _claude_bin():
    for p in (shutil.which("claude"), os.path.expanduser("~/.local/bin/claude"),
              os.path.expanduser("~/.claude/local/claude"), "/opt/homebrew/bin/claude"):
        if p and os.path.exists(p):
            return p
    raise LLMError("claude CLI não encontrado (instala o Claude Code)")


def _run_claude(prompt, model, timeout, web):
    cmd = [_claude_bin(), "-p", "--model", model, "--append-system-prompt", RULES]
    if web:
        cmd += ["--allowedTools", "WebSearch,WebFetch", "--max-turns", "12"]
    else:
        cmd += ["--max-turns", "1"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise LLMError((r.stderr or r.stdout or "erro")[-400:])
    return r.stdout


def _run_codex(prompt, model, timeout, web):
    exe = shutil.which("codex")
    if not exe:
        raise LLMError("codex CLI não encontrado")
    cmd = [exe, "exec", "--skip-git-repo-check"]
    if model and model not in ("default", "sonnet", "opus", "haiku"):
        cmd += ["-m", model]
    cmd.append(RULES + "\n\n" + prompt)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise LLMError((r.stderr or r.stdout or "erro")[-400:])
    return r.stdout


def ask(prompt: str, backend: str = "claude", model: str = "sonnet", web: bool = False,
        timeout: int = 300, retries: int = 1):
    if backend == "mock":
        return None
    runner = {"claude": _run_claude, "codex": _run_codex}.get(backend)
    if runner is None:
        raise LLMError(f"backend desconhecido: {backend}")
    last = None
    for _ in range(retries + 1):
        try:
            return _extract_json(runner(prompt, model, timeout, web))
        except (LLMError, subprocess.TimeoutExpired) as e:
            last = e
            prompt += "\n\nLEMBRETE: responde SÓ com um bloco ```json com um objeto JSON válido."
    raise LLMError(str(last))


def check(backend: str, model: str) -> dict:
    return ask('Responde com {"ok": true}.', backend, model, timeout=120, retries=0)
