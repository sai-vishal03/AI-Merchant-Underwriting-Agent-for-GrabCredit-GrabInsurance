"""Production-safe OpenRouter LLM client utilities.

This module provides a single reusable function for synchronous chat-completion
calls using `requests`, suitable for underwriting narratives and message
formatting flows.
"""

from __future__ import annotations

import os
from typing import Any


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT_SECONDS = 30


class OpenRouterClientError(RuntimeError):
    """Raised when OpenRouter request execution or parsing fails."""


def _get_openrouter_api_key() -> str:
    """Load OpenRouter API key from environment.

    Raises:
        OpenRouterClientError: If `OPENROUTER_API_KEY` is missing or empty.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterClientError(
            "OPENROUTER_API_KEY is not set. Please export a valid API key in environment variables."
        )
    return api_key


def _extract_assistant_content(response_json: dict[str, Any]) -> str:
    """Extract assistant message content from OpenRouter response payload.

    Raises:
        OpenRouterClientError: If the expected payload structure is missing.
    """
    try:
        choices = response_json["choices"]
        if not choices:
            raise KeyError("choices is empty")
        content = choices[0]["message"]["content"]
    except (KeyError, TypeError, IndexError) as exc:
        raise OpenRouterClientError(
            "OpenRouter response format was unexpected; could not extract assistant message content."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise OpenRouterClientError(
            "OpenRouter returned an empty assistant message content."
        )

    return content


def generate_llm_response(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> str:
    """Generate a synchronous LLM response via OpenRouter chat completions.

    Args:
        system_prompt: System instruction prompt.
        user_prompt: User prompt content.
        temperature: Sampling temperature for generation.

    Returns:
        Assistant message content string.

    Raises:
        OpenRouterClientError: For missing API key, transport/API failures,
            or malformed responses.
    """
    api_key = _get_openrouter_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEFAULT_OPENROUTER_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        import requests

        response = requests.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise OpenRouterClientError(
            f"Failed to reach OpenRouter API (or requests dependency unavailable): {exc}"
        ) from exc

    if response.status_code >= 400:
        body_preview = response.text[:500]
        raise OpenRouterClientError(
            f"OpenRouter API returned HTTP {response.status_code}: {body_preview}"
        )

    try:
        response_json = response.json()
    except ValueError as exc:
        raise OpenRouterClientError(
            "OpenRouter API returned a non-JSON response."
        ) from exc

    return _extract_assistant_content(response_json)
