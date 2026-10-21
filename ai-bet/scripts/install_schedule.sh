#!/bin/zsh
# Agenda o OCTO BET no macOS (launchd).
# Uso: ./scripts/install_schedule.sh [config.toml] [HH:MM ...]
# Exemplo: ./scripts/install_schedule.sh config.toml 10:15
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
CFG=${1:-config.toml}; shift || true
TIMES=("$@"); [ ${#TIMES[@]} -eq 0 ] && TIMES=(10:15)
NAME=$(basename "$CFG" .toml)
LABEL="com.octobet.$NAME"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/logs"
INTERVALS=""
for t in "${TIMES[@]}"; do
  H=${t%:*}; M=${t#*:}
  INTERVALS+="<dict><key>Hour</key><integer>$((10#$H))</integer><key>Minute</key><integer>$((10#$M))</integer></dict>"
done
CLAUDE_DIR=$(dirname "$(command -v claude 2>/dev/null || echo $HOME/.local/bin/claude)")
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>bet</string>
    <string>--config</string><string>$CFG</string><string>run</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$CLAUDE_DIR:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>StartCalendarInterval</key><array>$INTERVALS</array>
  <key>StandardOutPath</key><string>$ROOT/data/logs/$NAME.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/$NAME.err</string>
</dict></plist>
PL
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✅ Agendado ($NAME) às ${TIMES[*]} todos os dias. Logs em data/logs/"
echo "   Para remover: launchctl unload $PLIST && rm $PLIST"
