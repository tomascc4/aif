#!/bin/zsh
# OCTO BET — instalação num Mac (corre uma vez: ./setup.sh)
set -e
cd "$(dirname "$0")"
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v $c >/dev/null 2>&1; then
    v=$($c -c 'import sys; print(sys.version_info[:2] >= (3,11))')
    if [ "$v" = "True" ]; then PY=$c; break; fi
  fi
done
if [ -z "$PY" ]; then echo "❌ Precisas de Python 3.11 ou mais recente (brew install python@3.12)"; exit 1; fi
echo "🐍 A usar $($PY --version)"
$PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "✅ Instalado."
if command -v claude >/dev/null 2>&1 || [ -x "$HOME/.local/bin/claude" ]; then
  echo "✅ Claude Code encontrado (é o mesmo do ai-fund)."
else
  echo "⚠️  Não encontrei o Claude Code. Instala-o como fizeste no ai-fund."
fi
echo ""
echo "Próximos passos:"
echo "  .venv/bin/python -m bet demo        # ver o demo com ligas inventadas"
echo "  .venv/bin/python -m bet check       # testar dados reais + Claude"
echo "  .venv/bin/python -m bet backtest    # backtest com dados reais"
