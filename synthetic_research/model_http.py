"""HTTP JSON calls for the chat and embedding clients."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

_BUSY_STATUS = {429, 503}
_RETRY_WAITS = (1.5, 3.0)


def read_json_response(request_for, kind: str, label: str) -> dict:
    """POST once, and wait through a short busy response."""
    waits = (0.0, *_RETRY_WAITS)
    for attempt, pause in enumerate(waits):
        if pause:
            time.sleep(pause)
        try:
            with urllib.request.urlopen(request_for(), timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            busy = error.code in _BUSY_STATUS
            if busy and attempt < len(waits) - 1:
                print(f"[{kind}] busy, retry {attempt + 1}", flush=True)
                continue
            if busy:
                raise RuntimeError("The model is busy. Ask again in a moment.") from None
            raise RuntimeError(f"{label} {error.code}: {detail}") from None
    raise RuntimeError("The model is busy. Ask again in a moment.")
