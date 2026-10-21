#!/bin/zsh
# Deixa o dashboard sempre ligado em segundo plano (porta 8790).
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PLIST="$HOME/Library/LaunchAgents/com.octobet.dashboard.plist"
mkdir -p "$ROOT/data/logs"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.octobet.dashboard</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key><array><string>$ROOT/.venv/bin/python</string><string>-m</string><string>bet</string><string>dashboard</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/data/logs/dashboard.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/dashboard.err</string>
</dict></plist>
PL
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✅ Dashboard em http://localhost:8790"
