import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from synthetic_research.knowledge_mcp import store_path

HTTPS_URL = re.compile(r"https://[^\s)>\"]+")
_PAGE_LENGTH = re.compile(r"of (\d+) total")
PRIVATE_HOST = re.compile(
    r"^(localhost|.*\.local|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)$"
)


@dataclass(frozen=True)
class McpServer:
    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def _search_query(question: str, search_context: str | None) -> str:
    """Public search uses the respondent's role plus the question, capped for the tool."""
    role = " ".join(str(search_context or "").split())
    asked = " ".join(str(question or "").split())
    if role and role.lower() not in asked.lower():
        asked = f"{role} {asked}".strip()
    return asked[:400]


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _skipped(database: str, tool: str, error: str) -> dict:
    return {
        "database": database,
        "server": database,
        "tool": tool,
        "ok": False,
        "text": "",
        "url": "",
        "error": error,
    }


def _ok(database: str, server: str, tool: str, text: str, url: str = "") -> dict:
    return {
        "database": database,
        "server": server,
        "tool": tool,
        "ok": True,
        "text": " ".join(text.split())[:1200],
        "url": url,
        "error": "",
    }


def _failed(database: str, server: str, tool: str, error: Exception) -> dict:
    return _skipped(database, tool, str(error)[:180]) | {"server": server}


def public_https(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return False
    return PRIVATE_HOST.match(host) is None


def first_public_https(text: str) -> str:
    for match in HTTPS_URL.findall(text or ""):
        url = match.rstrip(".,;")
        if public_https(url):
            return url
    return ""


class ResearchMcp:
    """MCP client. Each database is an official server called with tools/call."""

    def gather(
        self,
        question: str,
        cohort_id: str,
        session_id: str | None = None,
        query_vector: list[float] | None = None,
        search_context: str | None = None,
    ) -> list[dict]:
        del cohort_id
        results = []
        if _enabled("MCP_WEB"):
            results.extend(self._web(_search_query(question, search_context)))
        else:
            results.append(_skipped("web", "search", "MCP_WEB is off."))
        if _enabled("MCP_VECTOR"):
            results.append(self._vector(question, session_id, query_vector))
        else:
            results.append(_skipped("vector", "vector_query", "MCP_VECTOR is off."))
        if _enabled("MCP_GRAPH"):
            results.append(self._graph(question, session_id))
        else:
            results.append(_skipped("graph", "graph_query", "MCP_GRAPH is off."))
        return results

    def stored(self, session_id: str) -> bool:
        directory = os.environ.get("KNOWLEDGE_MCP_DIR", "").strip()
        if not directory or not session_id:
            return True
        try:
            path = store_path(directory, session_id)
        except ValueError:
            return True
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(payload.get("vectors"))

    def index_knowledge(self, session_id: str, points: list[dict], graph: dict) -> list[dict]:
        if not os.environ.get("KNOWLEDGE_MCP_DIR", "").strip():
            return []
        results = []
        if _enabled("MCP_VECTOR"):
            results.append(
                self._knowledge(
                    "vector",
                    "vector_upsert",
                    {"session_id": session_id, "points_json": json.dumps(points)},
                )
            )
        if _enabled("MCP_GRAPH"):
            results.append(
                self._knowledge(
                    "graph",
                    "graph_upsert",
                    {
                        "session_id": session_id,
                        "graph_json": json.dumps(
                            {"nodes": graph.get("nodes") or [], "edges": graph.get("edges") or []}
                        ),
                    },
                )
            )
        return results

    def _web(self, question: str) -> list[dict]:
        key = os.environ.get("BRAVE_API_KEY", "").strip()
        if key:
            return self._brave(question)
        if _uvx_command() is None:
            return [
                _skipped(
                    "web",
                    "search",
                    "uvx is not installed; install uv (https://docs.astral.sh/uv/) or set BRAVE_API_KEY.",
                )
            ]
        return self._duckduckgo(question)

    def _duckduckgo(self, question: str) -> list[dict]:
        server = _duckduckgo_server()
        if server is None:
            return [
                _skipped(
                    "web",
                    "search",
                    "uvx is not installed; install uv (https://docs.astral.sh/uv/) or set BRAVE_API_KEY.",
                )
            ]
        try:
            raw = self.call_tool(
                server,
                "search",
                {
                    "query": question[:400],
                    "max_results": 3,
                    "region": os.environ.get("DDG_REGION", "in-en"),
                },
            )
        except Exception as error:
            return [_failed("web", "duckduckgo", "search", error)]
        results = [_ok("web", "duckduckgo", "search", raw, first_public_https(raw))]
        url = first_public_https(raw)
        if url:
            results.append(self._duckduckgo_fetch(url))
        return results

    def _duckduckgo_fetch(self, url: str) -> dict:
        server = _duckduckgo_server()
        if server is None:
            return _skipped(
                "web",
                "fetch_content",
                "uvx is not installed; install uv (https://docs.astral.sh/uv/) or set BRAVE_API_KEY.",
            )
        try:
            page = self.call_tool(
                server,
                "fetch_content",
                {"url": url, "max_length": 2000},
            )
        except Exception as error:
            return _failed("web", "duckduckgo", "fetch_content", error)
        readable = _PAGE_LENGTH.search(page or "")
        if readable and int(readable.group(1)) < 80:
            return _failed("web", "duckduckgo", "fetch_content", RuntimeError("The page had no readable text."))
        return _ok("web", "duckduckgo", "fetch_content", page, url)

    def _brave(self, question: str) -> list[dict]:
        key = os.environ.get("BRAVE_API_KEY", "").strip()
        try:
            raw = self.call_tool(
                _brave_server(key),
                "brave_web_search",
                {"query": question[:400], "count": 3, "country": "IN"},
            )
        except Exception as error:
            return [_failed("web", "brave-search", "brave_web_search", error)]
        results = [_ok("web", "brave-search", "brave_web_search", raw)]
        url = first_public_https(raw)
        if url:
            results.append(self._fetch(url))
        return results

    def _fetch(self, url: str) -> dict:
        server = _fetch_server()
        if server is None:
            return _skipped(
                "web",
                "fetch",
                "uvx is not installed; install uv (https://docs.astral.sh/uv/) or set BRAVE_API_KEY.",
            )
        try:
            page = self.call_tool(server, "fetch", {"url": url, "max_length": 2000})
        except Exception as error:
            return _failed("web", "fetch", "fetch", error)
        return _ok("web", "fetch", "fetch", page, url)

    def _vector(self, question: str, session_id: str | None, query_vector: list[float] | None) -> dict:
        if os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_LOCAL_PATH"):
            try:
                found = self.call_tool(_qdrant_server(), "qdrant-find", {"query": question[:400]})
            except Exception as error:
                return _failed("vector", "qdrant", "qdrant-find", error)
            return _ok("vector", "qdrant", "qdrant-find", found)
        if os.environ.get("KNOWLEDGE_MCP_DIR") and session_id:
            return self._knowledge(
                "vector",
                "vector_query",
                {
                    "session_id": session_id,
                    "question": question[:400],
                    "vector_json": json.dumps(query_vector) if query_vector else "",
                    "limit": 4,
                },
            )
        return _skipped("vector", "vector_query", "Set QDRANT_URL, QDRANT_LOCAL_PATH, or KNOWLEDGE_MCP_DIR.")

    def _graph(self, question: str, session_id: str | None) -> dict:
        if os.environ.get("KNOWLEDGE_MCP_DIR") and session_id and not os.environ.get("NEO4J_MCP_PASSWORD"):
            return self._knowledge(
                "graph",
                "graph_query",
                {"session_id": session_id, "question": question[:400]},
            )
        if not os.environ.get("NEO4J_MCP_PASSWORD"):
            return _skipped("graph", "graph_query", "Set KNOWLEDGE_MCP_DIR or NEO4J_MCP_PASSWORD.")
        try:
            schema = self.call_tool(_neo4j_server(), "get-schema", {})
        except Exception as error:
            return _failed("graph", "neo4j", "get-schema", error)
        return _ok("graph", "neo4j", "get-schema", schema)

    def _knowledge(self, database: str, tool: str, arguments: dict) -> dict:
        try:
            text = self.call_tool(_knowledge_server(), tool, arguments)
        except Exception as error:
            return _failed(database, "session-knowledge", tool, error)
        if database == "graph" and tool == "graph_query" and not str(text or "").strip():
            return _skipped("graph", "graph_query", "No graph stored or no edge matches this question.")
        return _ok(database, "session-knowledge", tool, text)

    def call_tool(self, server: McpServer, tool: str, arguments: dict) -> str:
        return asyncio.run(_call_tool(server, tool, arguments))


def _uvx_command() -> str | None:
    import shutil

    found = shutil.which("uvx")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "uvx"
    if local.is_file():
        return str(local)
    return None


def _duckduckgo_server() -> McpServer | None:
    command = _uvx_command()
    if not command:
        return None
    return McpServer(
        "duckduckgo",
        command,
        ("--with", "duckduckgo-mcp-server[browser]", "duckduckgo-mcp-server"),
        {
            "DDG_SAFE_SEARCH": os.environ.get("DDG_SAFE_SEARCH", "MODERATE"),
            "DDG_REGION": os.environ.get("DDG_REGION", "in-en"),
        },
    )


def _fetch_server() -> McpServer | None:
    command = _uvx_command()
    if not command:
        return None
    return McpServer("fetch", command, ("mcp-server-fetch",), {})


def _brave_server(key: str) -> McpServer:
    return McpServer(
        "brave-search",
        "npx",
        ("-y", "@brave/brave-search-mcp-server", "--transport", "stdio"),
        {"BRAVE_API_KEY": key},
    )


def _qdrant_server() -> McpServer:
    env = {"COLLECTION_NAME": os.environ.get("COLLECTION_NAME", "cam-1842")}
    for name in ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_LOCAL_PATH"):
        value = os.environ.get(name, "").strip()
        if value:
            env[name] = value
    return McpServer("qdrant", "uvx", ("mcp-server-qdrant",), env)


def _knowledge_server() -> McpServer:
    root = str(Path(__file__).resolve().parents[1])
    existing = os.environ.get("PYTHONPATH", "")
    return McpServer(
        "session-knowledge",
        sys_executable(),
        ("-m", "synthetic_research.knowledge_mcp"),
        {
            "KNOWLEDGE_MCP_DIR": os.environ.get("KNOWLEDGE_MCP_DIR", ""),
            "PYTHONPATH": root if not existing else root + os.pathsep + existing,
        },
    )


def enable_session_mcp(root: Path) -> None:
    """Turn on web search plus the local vector and graph MCP servers."""
    directory = root / "data" / "knowledge-mcp"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["KNOWLEDGE_MCP_DIR"] = str(directory)
    os.environ["MCP_WEB"] = "true"
    os.environ.setdefault("DDG_REGION", "in-en")
    if not os.environ.get("QDRANT_URL", "").strip() and not os.environ.get("QDRANT_LOCAL_PATH", "").strip():
        os.environ["MCP_VECTOR"] = "true"
    if not os.environ.get("NEO4J_MCP_PASSWORD", "").strip():
        os.environ["MCP_GRAPH"] = "true"


def _neo4j_server() -> McpServer:
    env = {
        "NEO4J_MCP_URI": os.environ.get("NEO4J_MCP_URI", "bolt://localhost:7687"),
        "NEO4J_MCP_USERNAME": os.environ.get("NEO4J_MCP_USERNAME", "neo4j"),
        "NEO4J_MCP_PASSWORD": os.environ.get("NEO4J_MCP_PASSWORD", ""),
        "NEO4J_MCP_DATABASE": os.environ.get("NEO4J_MCP_DATABASE", "neo4j"),
        "NEO4J_MCP_READ_ONLY": "true",
        "NEO4J_MCP_TELEMETRY": "false",
    }
    return McpServer("neo4j", sys_executable(), ("-m", "neo4j_mcp_server"), env)


def sys_executable() -> str:
    import sys

    return sys.executable


async def _call_tool(server: McpServer, tool: str, arguments: dict) -> str:
    from mcp import ClientSession
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=server.command,
        args=list(server.args),
        env={**os.environ, **server.env},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        raise RuntimeError(_result_text(result) or f"{tool} failed")
    text = _result_text(result)
    if not text:
        raise RuntimeError(f"{tool} returned an empty result")
    return text


def _result_text(result: object) -> str:
    content = getattr(result, "content", None) or []
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()
