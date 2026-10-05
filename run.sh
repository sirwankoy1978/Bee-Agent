#!/usr/bin/env bash
# Convenience launcher: sets up the virtualenv if needed and runs the bot.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing / upgrading dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> Starting Telegram AI bot..."
exec python bot.py
