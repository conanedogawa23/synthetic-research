"""Embedding calls for the session index.

The chat model does not serve the embedding model, so this client posts to
the configured embedding URL with a bearer token.
"""

from __future__ import annotations

import json
import urllib.request

from synthetic_research.model_http import read_json_response


class EmbedClient:
    def __init__(self, config: dict):
        self._config = config

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._config.get("embed_model") or ""
        if not model:
            raise RuntimeError("EMBED_MODEL is required for the session index.")
        body = json.dumps({"model": model, "input": texts}).encode("utf-8")
        embed_url = self._config.get("embed_url") or self._config["base_url"]
        payload = read_json_response(
            lambda: urllib.request.Request(
                f"{embed_url}/embeddings",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._config['api_key']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            ),
            "embed",
            "Embeddings",
        )
        rows = payload.get("data") or []
        rows.sort(key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") for row in rows]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise RuntimeError("The embedding model returned an incomplete batch.")
        return vectors
