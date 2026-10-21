import json
import os
from pathlib import Path

# embeddinggemma-300m is not a Bedrock model. The indexed sessions use this endpoint.
_EMBEDDINGGEMMA_URL = "https://api.deepinfra.com/v1/openai"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#") or "=" not in trimmed:
            continue
        key, value = trimmed.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not os.environ.get(key):
            os.environ[key] = value


def _setting(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def read_config(root: Path) -> dict:
    load_env_file(root / ".env")
    fast_model = _setting("CHAT_MODEL", "DEEPINFRA_FAST_MODEL")
    base_url = _setting("CHAT_URL", "DEEPINFRA_CHAT_URL").rstrip("/")
    embed_url = _setting("EMBED_URL", "DEEPINFRA_EMBED_URL").rstrip("/")
    if not embed_url:
        embed_url = (
            _EMBEDDINGGEMMA_URL
            if "bedrock-mantle." in base_url or "bedrock-runtime." in base_url
            else base_url
        )
    enabled = _setting("ENABLE_CHAT", "ENABLE_DEEPINFRA") == "true"
    return {
        "enabled": enabled,
        "base_url": base_url,
        "embed_url": embed_url,
        "api_key": _setting("EMBED_API_KEY", "DEEPINFRA_API_KEY"),
        "fast_model": fast_model,
        "reasoning_model": _setting("CHAT_REASONING_MODEL", "DEEPINFRA_REASONING_MODEL", default=fast_model),
        "embed_model": _setting("EMBED_MODEL", "DEEPINFRA_EMBED_MODEL", default="google/embeddinggemma-300m"),
    }


def assert_inference_config(config: dict) -> None:
    if not config["enabled"]:
        raise RuntimeError("ENABLE_CHAT must be true.")
    missing = [
        name
        for name in ("base_url", "api_key", "fast_model", "reasoning_model")
        if not config[name]
    ]
    if missing:
        raise RuntimeError("Chat URL, embedding API key, and both model names are required.")


def load_pack(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
