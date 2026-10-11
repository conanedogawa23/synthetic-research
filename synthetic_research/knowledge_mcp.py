"""stdio MCP server for one session's vectors and graph.

The research process launches this with ``python -m synthetic_research.knowledge_mcp``.
Qdrant and Neo4j stay available when those servers are configured; this server is
the store the interview can actually read and write on this machine.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,79}$")
_TOKEN = re.compile(r"[a-z0-9]+")


def store_path(directory: str, session_id: str) -> Path:
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("invalid session id")
    root = Path(directory).resolve()
    return root / f"{session_id}.json"


def _directory() -> str:
    directory = os.environ.get("KNOWLEDGE_MCP_DIR", "").strip()
    if not directory:
        raise ValueError("KNOWLEDGE_MCP_DIR is not set.")
    return directory


def _read(session_id: str) -> dict:
    path = store_path(_directory(), session_id)
    if not path.is_file():
        return {"vectors": [], "graph": {"nodes": [], "edges": []}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("knowledge store is not an object")
    payload.setdefault("vectors", [])
    payload.setdefault("graph", {"nodes": [], "edges": []})
    return payload


def _write(session_id: str, payload: dict) -> None:
    path = store_path(_directory(), session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def vector_upsert(session_id: str, points_json: str) -> str:
    """Replace the vectors stored for one session."""
    points = json.loads(points_json)
    if not isinstance(points, list):
        raise ValueError("points_json must be a list")
    kept = []
    for point in points:
        if not isinstance(point, dict) or not point.get("id") or not isinstance(point.get("vector"), list):
            continue
        kept.append(
            {
                "id": str(point["id"]),
                "text": str(point.get("text") or "")[:2000],
                "vector": [float(value) for value in point["vector"]],
            }
        )
    payload = _read(session_id)
    payload["vectors"] = kept
    _write(session_id, payload)
    return f"Stored {len(kept)} vectors for {session_id}."


def vector_query(session_id: str, question: str, vector_json: str = "", limit: int = 4) -> str:
    """Return the closest stored passages for a question."""
    points = _read(session_id).get("vectors") or []
    if not points:
        raise ValueError("No vectors are stored for this session.")
    query = _vector(vector_json)
    ranked = []
    for point in points:
        score = _cosine(query, point.get("vector") or []) if query else _overlap(question, point.get("text") or "")
        if score <= 0:
            continue
        ranked.append((score, point))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        raise ValueError("No stored passage matches this question.")
    lines = []
    for score, point in ranked[: max(1, min(int(limit), 8))]:
        text = " ".join(str(point.get("text") or "").split())
        lines.append(f"{point.get('id')} ({score:.2f}) {text[:320]}")
    return "\n".join(lines)


def graph_upsert(session_id: str, graph_json: str) -> str:
    """Replace the entity graph stored for one session."""
    graph = json.loads(graph_json)
    if not isinstance(graph, dict):
        raise ValueError("graph_json must be an object")
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    payload = _read(session_id)
    payload["graph"] = {"nodes": nodes, "edges": edges}
    _write(session_id, payload)
    return f"Stored {len(nodes)} nodes and {len(edges)} edges for {session_id}."


def graph_query(session_id: str, question: str) -> str:
    """Return graph edges whose labels share words with the question."""
    edges = (_read(session_id).get("graph") or {}).get("edges") or []
    if not edges:
        raise ValueError("No graph is stored for this session.")
    question_tokens = set(_TOKEN.findall(question.lower()))
    ranked = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        label = f"{edge.get('from') or ''} {edge.get('relation') or ''} {edge.get('to') or ''}"
        overlap = len(question_tokens & set(_TOKEN.findall(label.lower())))
        if overlap:
            ranked.append((overlap, label))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        raise ValueError("No graph edge matches this question.")
    return "\n".join(label for _, label in ranked[:8])


def _vector(vector_json: str) -> list[float]:
    if not str(vector_json or "").strip():
        return []
    parsed = json.loads(vector_json)
    if not isinstance(parsed, list):
        return []
    return [float(value) for value in parsed]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for first, second in zip(left, right):
        dot += first * second
        left_norm += first * first
        right_norm += second * second
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _overlap(question: str, text: str) -> float:
    question_tokens = set(_TOKEN.findall(question.lower()))
    text_tokens = set(_TOKEN.findall(text.lower()))
    if not question_tokens or not text_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / len(question_tokens)


def main() -> None:
    from mcp.server import MCPServer

    server = MCPServer("session-knowledge", log_level="WARNING")
    server.tool()(vector_upsert)
    server.tool()(vector_query)
    server.tool()(graph_upsert)
    server.tool()(graph_query)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
