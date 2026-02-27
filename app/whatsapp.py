"""WhatsApp delivery adapter for underwriting offers.

Supports a safe mock mode by default and optional Twilio-backed delivery when
credentials are provided.
"""

from __future__ import annotations

import os
import random
from typing import Any


class WhatsAppDeliveryError(RuntimeError):
    """Raised when WhatsApp delivery fails in provider mode."""


def _mock_send(merchant_id: str, phone_number: str, message: str) -> dict[str, Any]:
    """Return deterministic mock payload for demo/sandbox usage."""
    return {
        "merchant_id": merchant_id,
        "phone_number": phone_number,
        "provider": "mock",
        "status": "queued",
        "message_id": f"MOCK-WA-{random.randint(100000, 999999)}",
        "preview": message[:160],
    }


def _twilio_send(merchant_id: str, phone_number: str, message: str) -> dict[str, Any]:
    """Send WhatsApp notification using Twilio REST API via requests."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", "").strip()

    if not account_sid or not auth_token or not from_number:
        raise WhatsAppDeliveryError(
            "Twilio credentials are missing. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM."
        )

    try:
        import requests

        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            data={
                "From": from_number,
                "To": phone_number,
                "Body": message,
            },
            auth=(account_sid, auth_token),
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise WhatsAppDeliveryError(f"Twilio request failed: {exc}") from exc

    if response.status_code >= 400:  # pragma: no cover - external API path
        raise WhatsAppDeliveryError(
            f"Twilio API error {response.status_code}: {response.text[:300]}"
        )

    payload = response.json()
    return {
        "merchant_id": merchant_id,
        "phone_number": phone_number,
        "provider": "twilio",
        "status": payload.get("status", "queued"),
        "message_id": payload.get("sid", ""),
        "preview": message[:160],
    }


def send_whatsapp_offer(merchant_id: str, phone_number: str, message: str) -> dict[str, Any]:
    """Send offer notification via configured provider.

    Provider selection:
    - `WHATSAPP_PROVIDER=twilio` => Twilio API
    - default => mock mode
    """
    provider = os.getenv("WHATSAPP_PROVIDER", "mock").strip().lower()

    if provider == "twilio":
        return _twilio_send(merchant_id=merchant_id, phone_number=phone_number, message=message)

    return _mock_send(merchant_id=merchant_id, phone_number=phone_number, message=message)
