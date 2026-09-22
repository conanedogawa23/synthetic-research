import json
import os
from pathlib import Path


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


def read_config(root: Path) -> dict:
    load_env_file(root / ".env")
    fast_model = os.environ.get("DEEPINFRA_FAST_MODEL", "")
    return {
        "enabled": os.environ.get("ENABLE_DEEPINFRA") == "true",
        "base_url": os.environ.get("DEEPINFRA_CHAT_URL", "").rstrip("/"),
        "api_key": os.environ.get("DEEPINFRA_API_KEY", ""),
        "fast_model": fast_model,
        "reasoning_model": os.environ.get("DEEPINFRA_REASONING_MODEL", fast_model),
        "embed_model": os.environ.get("DEEPINFRA_EMBED_MODEL", "google/embeddinggemma-300m"),
    }


def assert_inference_config(config: dict) -> None:
    if not config["enabled"]:
        raise RuntimeError("ENABLE_DEEPINFRA must be true.")
    missing = [
        name
        for name in ("base_url", "api_key", "fast_model", "reasoning_model")
        if not config[name]
    ]
    if missing:
        raise RuntimeError("DeepInfra URL, key, and both model names are required.")


def load_pack(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
