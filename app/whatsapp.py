"""Twilio WhatsApp sandbox utility for offer notifications."""

from __future__ import annotations

import base64
import os
import urllib.parse
import urllib.request


def send_whatsapp_offer(phone_number: str, message: str) -> dict[str, str | bool]:
    """Send a WhatsApp offer message through Twilio sandbox credentials.

    Returns a structured result without raising hard failures when credentials are absent.
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

    if not sid or not token or not from_number:
        return {"sent": False, "reason": "twilio_credentials_missing"}

    payload = urllib.parse.urlencode(
        {
            "To": f"whatsapp:{phone_number}" if not phone_number.startswith("whatsapp:") else phone_number,
            "From": from_number,
            "Body": message,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=payload,
        method="POST",
    )
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("utf-8")
    req.add_header("Authorization", f"Basic {auth}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=10):
            return {"sent": True, "reason": "ok"}
    except Exception:
        return {"sent": False, "reason": "twilio_send_failed"}
