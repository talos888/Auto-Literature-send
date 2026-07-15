#!/usr/bin/env python3
"""Send the weekly literature radar report by SMTP.

All credentials must come from environment variables. The script intentionally
does not support credentials in config files or command-line arguments.
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


SECRET_ARG_PATTERNS = ("password", "passwd", "secret", "token", "api_key", "apikey")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def build_message(report_path: Path, subject: str, sender: str, recipient: str) -> EmailMessage:
    report = report_path.read_text(encoding="utf-8")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(report)
    msg.add_attachment(
        report,
        subtype="markdown",
        filename=report_path.name,
    )
    return msg


def send_message(msg: EmailMessage, secrets: list[str]) -> None:
    host = require_env("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT") or "587")
    username = os.environ.get("SMTP_USERNAME", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    use_ssl = os.environ.get("SMTP_SSL", "false").lower() in {"1", "true", "yes"}

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as server:
                if username or password:
                    server.login(username, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=60) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                if username or password:
                    server.login(username, password)
                server.send_message(msg)
    except Exception as exc:
        raise RuntimeError(redact(str(exc), secrets)) from exc


def main() -> int:
    if any(pattern in arg.lower() for arg in sys.argv[1:] for pattern in SECRET_ARG_PATTERNS):
        raise RuntimeError("Do not pass credentials on the command line. Use environment variables only.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--subject", default="Weekly arXiv Literature Radar")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report)
    sender = require_env("MAIL_FROM")
    recipient = require_env("MAIL_TO")
    secrets = [
        os.environ.get("SMTP_PASSWORD", ""),
        os.environ.get("SMTP_USERNAME", ""),
        os.environ.get("SMTP_HOST", ""),
    ]

    msg = build_message(report_path, args.subject, sender, recipient)
    if args.dry_run:
        print(f"Dry run: would send {report_path} to {recipient}")
        return 0

    send_message(msg, secrets)
    print(f"Sent {report_path} to {recipient}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
