"""Ollama embeddings client with on-disk caching.

We use this to upgrade the TF-IDF persona-retention and language-similarity
metrics. Embeddings are 50-100x cheaper than generation, so we batch
aggressively but a single call is fine for the experiment scale.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text",
                 ollama_url: str = "http://localhost:11434",
                 cache_dir: Optional[str] = None,
                 timeout: int = 60):
        self.model = model
        self.url = f"{ollama_url}/api/embeddings"
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, np.ndarray] = {}
        self._session = requests.Session()

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(
            (self.model + "\0" + text).encode("utf-8")
        ).hexdigest()

    def _disk_get(self, key: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.npy"
        if p.exists():
            try:
                return np.load(p)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _disk_put(self, key: str, v: np.ndarray) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / f"{key}.npy"
        try:
            np.save(p, v)
        except Exception:  # noqa: BLE001
            pass

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(768, dtype=np.float32)
        key = self._cache_key(text)
        if key in self._mem_cache:
            return self._mem_cache[key]
        cached = self._disk_get(key)
        if cached is not None:
            self._mem_cache[key] = cached
            return cached
        r = self._session.post(self.url, json={
            "model": self.model,
            "prompt": text,
        }, timeout=self.timeout)
        r.raise_for_status()
        v = np.asarray(r.json().get("embedding", []), dtype=np.float32)
        if v.size == 0:
            v = np.zeros(768, dtype=np.float32)
        # L2 normalize so dot product = cosine
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        self._mem_cache[key] = v
        self._disk_put(key, v)
        return v

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [self.embed(t) for t in texts]
        return np.vstack(rows)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        # vectors are L2-normalized → dot = cosine
        return float(np.dot(a, b))
