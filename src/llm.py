"""Minimal Ollama client with JSON-mode helper."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class LLMConfig:
    model: str
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.8
    max_tokens: int = 220
    request_timeout: int = 120


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._session = requests.Session()

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False, temperature: Optional[float] = None) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            # Disable reasoning for reasoning-capable models (qwen3.x, etc.).
            # Their default output goes into a separate `thinking` field and
            # `response` comes back empty until the model "decides" to answer,
            # which can blow through num_predict before anything is emitted.
            "think": False,
            "options": {
                "temperature": temperature if temperature is not None else self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        url = f"{self.cfg.ollama_url}/api/generate"
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                r = self._session.post(url, json=payload, timeout=self.cfg.request_timeout)
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama generate failed: {last_err}")

    def generate_json(self, prompt: str, system: Optional[str] = None,
                      temperature: Optional[float] = None,
                      retries: int = 2) -> dict[str, Any]:
        """Generate a response and parse it as JSON, with light repair."""
        last_text = ""
        for _ in range(retries + 1):
            text = self.generate(prompt, system=system, json_mode=True, temperature=temperature)
            last_text = text
            parsed = _safe_json_parse(text)
            if parsed is not None:
                return parsed
        # Fallback: return raw text under a key so caller can still log it.
        return {"_parse_error": True, "raw": last_text}


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _safe_json_parse(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    # First try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass
    # Try to extract the largest {...} substring
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            return None
    return None
