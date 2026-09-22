import json
from pathlib import Path


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.sessions = root / "sessions"
        self.vault = root / "vault"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.vault.mkdir(parents=True, exist_ok=True)

    def _dir(self, session_id: str) -> Path:
        return self.sessions / session_id

    def create(self, session: dict) -> dict:
        directory = self._dir(session["sessionId"])
        directory.mkdir(parents=True, exist_ok=True)
        self.write(session)
        return session

    def read(self, session_id: str) -> dict:
        path = self._dir(session_id) / "session.json"
        if not path.exists():
            from synthetic_research.gates import ResearchError

            raise ResearchError("SESSION_MISSING", f"Session {session_id} was not found.")
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, session: dict) -> dict:
        path = self._dir(session["sessionId"]) / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
        return session

    def write_index(self, session_id: str, payload: dict) -> None:
        path = self._dir(session_id) / "index.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def read_index(self, session_id: str) -> dict | None:
        path = self._dir(session_id) / "index.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def drop_index(self, session_id: str) -> None:
        path = self._dir(session_id) / "index.json"
        if path.exists():
            path.unlink()

    def index_exists(self, session_id: str) -> bool:
        return (self._dir(session_id) / "index.json").exists()

    def promote_pack(self, session_id: str, pack: dict) -> Path:
        path = self.vault / session_id / "pack.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        return path

    def read_vault(self, session_id: str) -> dict | None:
        path = self.vault / session_id / "pack.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
