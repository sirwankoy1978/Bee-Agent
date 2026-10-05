"""
Optional helper to register a Telegram webhook (only needed if you want
to run behind a public HTTPS URL instead of using the built-in polling mode).

Usage:
    python set_webhook.py https://your-subdomain.ngrok-free.app
    python set_webhook.py --info
    python set_webhook.py --delete
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET_TOKEN", "")

if not TELEGRAM_TOKEN:
    sys.exit("TELEGRAM_TOKEN missing")

API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url", nargs="?")
    p.add_argument("--info", action="store_true")
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()

    if args.info:
        r = requests.post(f"{API}/getWebhookInfo", timeout=15)
        print(r.text)
        return
    if args.delete:
        r = requests.post(f"{API}/deleteWebhook", data={"drop_pending_updates": True}, timeout=15)
        print(r.text)
        return
    if not args.url:
        p.print_help()
        return

    webhook = args.url.rstrip("/") + "/telegram/webhook"
    secret = SECRET or secrets.token_urlsafe(32)
    r = requests.post(
        f"{API}/setWebhook",
        data={"url": webhook, "secret_token": secret, "drop_pending_updates": True},
        timeout=15,
    )
    print(r.text)


if __name__ == "__main__":
    main()
