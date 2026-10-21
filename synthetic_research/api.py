"""HTTP adapter so Campaign OS can drive a research session without the terminal."""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from synthetic_research.chat import ChatClient
from synthetic_research.config import assert_inference_config, read_config
from synthetic_research.corpus import validate_campaign
from synthetic_research.embeddings import EmbedClient
from synthetic_research.engine import TURN_CAP, ResearchEngine
from synthetic_research.gates import ResearchError
from synthetic_research.grounding import visible_evidence_ids
from synthetic_research.mcp_client import ResearchMcp, enable_session_mcp
from synthetic_research.s3_log import read_session_log
from synthetic_research.store import SessionStore
from synthetic_research.synthesis import plan_synthesis, public_synthesis

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_KINDS = {"transcript", "review", "screener", "sales", "web"}
SESSION_LOG = re.compile(r"^/sessions/([A-Za-z0-9][A-Za-z0-9_-]{7,80})\.json$")
PHASES = ("rapport", "unprompted", "deep_dive", "prompted")


def load_packs(gold_dir: Path) -> dict[str, dict]:
    packs = {}
    if not gold_dir.exists():
        return packs
    for path in sorted(gold_dir.glob("*.json")):
        pack = json.loads(path.read_text(encoding="utf-8"))
        campaign_id = pack.get("campaignId")
        if campaign_id:
            packs[campaign_id] = pack
    return packs


def _stamp(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, tail = text.split(".", 1)
        sign = max(tail.rfind("+"), tail.rfind("-"))
        offset = tail[sign:] if sign > 0 else "+00:00"
        return head + offset
    if "+" not in text[10:] and not text.endswith("+00:00"):
        return text + "+00:00"
    return text


PERSONA_TEMPLATES = [
    {
        "id": "skeptic",
        "label": "The Skeptic",
        "description": "Demands rigorous proof, dislikes buzzwords, and is highly critical of theoretical gains.",
        "prompt": "A high-level executive who is deeply skeptical. They prioritize hard evidence, specific figures, and historical failures over promised future gains."
    },
    {
        "id": "visionary",
        "label": "The Visionary",
        "description": "Focuses on long-term strategic shifts and industry-wide transformation.",
        "prompt": "A forward-thinking leader focused on the 5-year horizon. They are interested in how this change transforms the entire category, not just marginal gains."
    },
    {
        "id": "operator",
        "label": "The Operator",
        "description": "Concerned with day-to-day friction, field execution, and 'boots on the ground' reality.",
        "prompt": "A pragmatic leader focused on execution. They care about the friction of daily operations and how a solution actually works in the field."
    }
]

class ResearchApi:
    def __init__(self, engine: ResearchEngine, packs: dict[str, dict], campaign_dir: Path | None = None):
        self.engine = engine
        self.packs = dict(packs)
        self.campaign_dir = campaign_dir

    def handle(self, method: str, path: str, body: dict | None, actor: str) -> dict:
        parts = [part for part in urlparse(path).path.split("/") if part]
        try:
            data = self._route(method.upper(), parts, body or {}, actor or "Meera")
        except ResearchError as error:
            return {"ok": False, "code": error.code, "message": str(error)}
        except KeyError as error:
            return {"ok": False, "code": "PACK_INVALID", "message": f"Missing {error}."}
        return {"ok": True, "data": data}

    def _route(self, method: str, parts: list[str], body: dict, actor: str):
        if parts == ["health"] and method == "GET":
            return {"status": "ok"}
        if parts == ["personas", "templates"] and method == "GET":
            return PERSONA_TEMPLATES
        if parts == ["campaigns"] and method == "GET":
            self._refresh_saved()
            return [_campaign(pack) for pack in self.packs.values()]
        if parts == ["campaigns"] and method == "POST":
            return self._create_campaign(body)
        if len(parts) == 2 and parts[0] == "campaigns" and method == "GET":
            return _campaign(self._pack(parts[1]))
        if len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "kit" and method == "PATCH":
            return self._update_persona(parts[1], body)
        if len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "impact" and method == "GET":
            return self._persona_impact(parts[1])
        if len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "sessions" and method == "GET":
            self._pack(parts[1])
            return [_session(row) for row in self._sessions() if row.get("campaignId") == parts[1]]
        if len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "sessions" and method == "POST":
            session = self.engine.open(self._pack(parts[1]), actor=actor)
            return _session(session)
        if len(parts) == 2 and parts[0] == "sessions" and method == "GET":
            return _session(self.engine.store.read(parts[1]))
        if parts[:1] == ["sessions"] and len(parts) >= 3:
            return self._session_route(method, parts, body, actor)
        if parts[:1] == ["interviews"] and len(parts) >= 2:
            return self._interview_route(method, parts, body)
        raise ResearchError("NOT_FOUND", "That research route does not exist.")

    def _session_route(self, method: str, parts: list[str], body: dict, actor: str):
        session_id = parts[1]
        if len(parts) == 3 and parts[2] == "knowledge" and method == "GET":
            return self.engine.list_knowledge(session_id)
        if len(parts) == 3 and parts[2] == "knowledge" and method == "POST":
            filename = str(body.get("filename") or "").strip()
            encoded = body.get("contentBase64") or ""
            if not filename or not encoded:
                raise ResearchError("KNOWLEDGE_EMPTY", "Choose a file to upload.")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except Exception as error:  # noqa: BLE001
                raise ResearchError("PACK_INVALID", "The upload was not valid base64.") from error
            if len(payload) > 50 * 1024 * 1024:
                raise ResearchError("TOO_LARGE", "That file is over 50 MB.")
            return self.engine.add_knowledge(session_id, filename, payload, actor)
        if len(parts) == 4 and parts[2] == "knowledge" and parts[3] == "prepare" and method == "POST":
            steward = str(body.get("steward") or actor).strip()
            prepared = self.engine.prepare_knowledge(session_id, steward)
            return _session(prepared)
        if len(parts) == 3 and parts[2] == "grant" and method == "POST":
            steward = str(body.get("steward") or "").strip()
            if not steward:
                raise ResearchError("STEWARD_MISSING", "Name the steward granting this evidence.")
            pack = self._pack_for_session(session_id)
            granted = self.engine.grant(session_id, pack["grantId"], steward=steward)
            return _session(granted)
        if len(parts) == 3 and parts[2] == "cohorts" and method == "GET":
            session = self.engine.store.read(session_id)
            rows = self.engine.list_cohorts(session_id)
            labels = {rule["id"]: rule.get("label") or rule["id"] for rule in session.get("cohortRules") or []}
            return [
                {"id": row["id"], "label": labels.get(row["id"], row["id"]), "members": row["members"]}
                for row in rows
            ]
        if len(parts) == 3 and parts[2] == "respondents" and method == "GET":
            cards = self.engine.compile_cards(session_id)
            return [
                {"id": card["cardId"], "cohortId": card["cohortId"], "label": card["label"]}
                for card in cards
            ]
        if len(parts) == 3 and parts[2] == "interviews" and method == "GET":
            session = self.engine.store.read(session_id)
            return [_interview(session, interview) for interview in session.get("interviews") or []]
        if len(parts) == 3 and parts[2] == "interviews" and method == "POST":
            card_id = str(body.get("cardId") or "").strip()
            session = self.engine.store.read(session_id)
            cohort_id = next(
                (card["cohortId"] for card in session.get("cards") or [] if card["cardId"] == card_id),
                "",
            )
            if not cohort_id:
                raise ResearchError("CARD_MISSING", "Compile a card before starting an interview.")
            interview = self.engine.start_interview(session_id, cohort_id)
            return _interview(self.engine.store.read(session_id), interview)
        if len(parts) == 3 and parts[2] == "skip" and method == "POST":
            cohort_id = str(body.get("cohortId") or "").strip()
            reason = str(body.get("reason") or "").strip()
            self.engine.skip(session_id, cohort_id, reason)
            return _session(self.engine.store.read(session_id))
        if len(parts) == 3 and parts[2] == "pack" and method == "POST":
            return _pack_view(self.engine.draft_pack(session_id))
        if len(parts) == 3 and parts[2] == "pack" and method == "GET":
            session = self.engine.store.read(session_id)
            if not session.get("pack"):
                raise ResearchError("PACK_MISSING", "There is no insight pack yet.")
            return _pack_view(session["pack"])
        if len(parts) == 3 and parts[2] == "approve" and method == "POST":
            name = str(body.get("oversightName") or "").strip()
            vault = self.engine.approve(session_id, name)
            session = self.engine.store.read(session_id)
            view = _pack_view(session["pack"])
            view["oversightName"] = vault.get("oversightName")
            view["status"] = "approved"
            return view
        if len(parts) == 4 and parts[2] == "evidence" and method == "GET":
            chip = self.engine.open_evidence(session_id, parts[3])
            session = self.engine.store.read(session_id)
            kind = chip.get("kind") or "transcript"
            return {
                "spanId": chip["id"],
                "kind": kind if kind in EVIDENCE_KINDS else "transcript",
                "text": chip.get("text") or "",
                "source": chip.get("source") or kind,
                "market": (session.get("marketPack") or {}).get("market") or "",
            }
        if len(parts) >= 4 and parts[2] == "interviews":
            return self._nested_interview(method, session_id, parts[3], parts[4:], body)
        raise ResearchError("NOT_FOUND", "That research route does not exist.")

    def _nested_interview(self, method: str, session_id: str, interview_id: str, rest: list[str], body: dict):
        session = self.engine.store.read(session_id)
        interview = _match_interview(session, interview_id)
        if rest == [] and method == "GET":
            return _interview(session, interview)
        cohort_id = interview["cohortId"]
        if rest == ["questions"] and method == "POST":
            question = str(body.get("question") or "").strip()
            if not question:
                raise ResearchError("QUESTION_EMPTY", "Type a question before asking.")
            turn = self.engine.ask(session_id, cohort_id, question)
            follow = None
            try:
                follow = self.engine.follow_up_question(session_id, cohort_id, turn)
            except (ResearchError, RuntimeError):
                follow = None
            fresh = self.engine.store.read(session_id)
            return _turn(fresh, turn, follow)
        if rest == ["phase"] and method == "POST":
            updated = self.engine.advance_phase(session_id, cohort_id)
            return _interview(self.engine.store.read(session_id), updated)
        if rest == ["end"] and method == "POST":
            updated = self.engine.end_interview(session_id, cohort_id)
            return _interview(self.engine.store.read(session_id), updated)
        if rest == ["improve"] and method == "POST":
            instruction = str(body.get("instruction") or "").strip() or "Tighten this reply."
            turn = self.engine.improve(session_id, cohort_id, instruction)
            fresh = self.engine.store.read(session_id)
            return _turn(fresh, turn, None)
        raise ResearchError("NOT_FOUND", "That research route does not exist.")

    def _interview_route(self, method: str, parts: list[str], body: dict):
        interview_id = parts[1]
        session = self._session_for_interview(interview_id)
        return self._nested_interview(method, session["sessionId"], interview_id, parts[2:], body)

    def _update_persona(self, campaign_id: str, body: dict) -> dict:
        persona = str(body.get("persona") or "").strip()
        if not persona:
            raise ResearchError("PACK_INVALID", "A persona needs text.")
        pack = self._pack(campaign_id)
        pack.setdefault("kit", {})["persona"] = persona
        self.packs[campaign_id] = pack
        self._persist_pack(campaign_id, pack)
        open_sessions = self._apply_persona_to_open_sessions(campaign_id, persona)
        warning = None
        if open_sessions:
            warning = (
                f"The next question in {open_sessions} open session"
                f"{'' if open_sessions == 1 else 's'} will use this persona. "
                "Answers already stored stay as they are."
            )
        return {"persona": persona, "openSessions": open_sessions, "warning": warning}

    def _persona_impact(self, campaign_id: str) -> dict:
        self._pack(campaign_id)
        sessions = [row for row in self._sessions() if row.get("campaignId") == campaign_id]
        open_sessions = sum(1 for row in sessions if row.get("status") == "open")
        warning = None
        if open_sessions:
            warning = (
                f"Saving a new persona will apply to the next question in {open_sessions} open session"
                f"{'' if open_sessions == 1 else 's'}. Answers already stored stay as they are."
            )
        return {
            "openSessions": open_sessions,
            "totalSessions": len(sessions),
            "warning": warning,
        }

    def _persist_pack(self, campaign_id: str, pack: dict) -> None:
        if self.campaign_dir is None:
            return
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        path = self.campaign_dir / f"{campaign_id}.json"
        path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")

    def _apply_persona_to_open_sessions(self, campaign_id: str, persona: str) -> int:
        updated = 0
        for row in self._sessions():
            if row.get("campaignId") != campaign_id or row.get("status") != "open":
                continue
            kit = row.setdefault("kit", {})
            kit["persona"] = persona
            self.engine.store.write(row)
            updated += 1
        return updated

    def _refresh_saved(self) -> None:
        if self.campaign_dir is None:
            return
        for campaign_id, pack in load_packs(self.campaign_dir).items():
            self.packs[campaign_id] = pack

    def _create_campaign(self, body: dict) -> dict:
        name = str(body.get("name") or "").strip()
        market = str(body.get("market") or "").strip()
        category = str(body.get("category") or "").strip()
        if not name or not market or not category:
            raise ResearchError("PACK_INVALID", "A campaign needs a name, a market, and a category.")
        pack = _new_pack(name, market, category, set(self.packs))
        validate_campaign(pack)
        self.packs[pack["campaignId"]] = pack
        if self.campaign_dir is not None:
            self.campaign_dir.mkdir(parents=True, exist_ok=True)
            path = self.campaign_dir / f"{pack['campaignId']}.json"
            path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        return _campaign(pack)

    def _pack(self, campaign_id: str) -> dict:
        pack = self.packs.get(campaign_id)
        if not pack:
            raise ResearchError("NOT_FOUND", f"No campaign with id {campaign_id}.")
        return pack

    def _pack_for_session(self, session_id: str) -> dict:
        session = self.engine.store.read(session_id)
        return self._pack(session["campaignId"])

    def _sessions(self) -> list[dict]:
        root = self.engine.store.sessions
        rows = []
        if not root.exists():
            return rows
        for directory in root.iterdir():
            path = directory / "session.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        rows.sort(key=lambda row: row.get("openedAt") or "", reverse=True)
        return rows

    def _session_for_interview(self, interview_id: str) -> dict:
        for session in self._sessions():
            if _match_interview(session, interview_id, missing_ok=True):
                return session
        raise ResearchError("INTERVIEW_MISSING", f"No interview with id {interview_id}.")


def _new_pack(name: str, market: str, category: str, taken: set[str]) -> dict:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "campaign"
    campaign_id = slug
    while campaign_id in taken:
        campaign_id = f"{slug}-{uuid.uuid4().hex[:4]}"
    cohort_id = "respondents"
    return {
        "campaignId": campaign_id,
        "grantId": f"grant-{campaign_id}",
        "market": market,
        "category": category,
        "kit": {
            "version": "v1",
            "locked": True,
            "brandAlias": "KitName",
            "purpose": name,
        },
        "marketPack": {
            "market": market,
            "household": "One compiled respondent for this campaign.",
            "taboos": ["Do not invent facts that are not in the uploaded knowledge."],
        },
        "cohorts": [
            {
                "id": cohort_id,
                "cardId": "R1",
                "label": "Respondent",
                "rule": f"{name} · uploaded knowledge",
                "bounds": [category, market],
                "traits": ["uploaded knowledge"],
                "tension": "what the files say / what is still unstated",
                "affinity": [category],
            }
        ],
        "guide": {
            "mains": [
                {"id": "m1", "phase": "rapport", "text": "Walk me through how this comes up for you."},
                {"id": "m2", "phase": "unprompted", "text": "What have you tried and dropped?"},
                {"id": "m3", "phase": "deep_dive", "text": "Why does that matter?"},
                {"id": "m4", "phase": "prompted", "text": "What would you need to see before you would support this?"},
            ],
            "probes": [
                {"id": "p1", "text": "Why does that matter?", "when": "A problem is named"},
            ],
        },
        "documents": [
            {
                "id": "SC-01",
                "kind": "screener",
                "cohortIds": [cohort_id],
                "members": 1,
                "title": "Respondent",
                "text": f"One compiled respondent for {name}. Member count in this gold slice is 1. This is not a population claim.",
            }
        ],
    }


def _campaign(pack: dict) -> dict:
    purpose = str((pack.get("kit") or {}).get("purpose") or pack.get("campaignId") or "Campaign")
    sentence = purpose.split(". ")[0].strip().rstrip(".")
    return {
        "id": pack["campaignId"],
        "name": sentence[:120],
        "market": (pack.get("marketPack") or {}).get("market") or pack.get("market") or "",
        "category": pack.get("category") or "",
        "kitVersion": (pack.get("kit") or {}).get("version") or "",
        "persona": str((pack.get("kit") or {}).get("persona") or ""),
        "guide": [
            {"id": main.get("id") or f"m{index}", "phase": main.get("phase"), "text": main.get("text")}
            for index, main in enumerate((pack.get("guide") or {}).get("mains") or [], start=1)
            if main.get("phase") in PHASES and main.get("text")
        ],
    }


def _session(session: dict) -> dict:
    grant = session.get("grant") or {}
    active = grant.get("status") == "active" and session.get("status") != "closed"
    phase = None
    for interview in reversed(session.get("interviews") or []):
        if interview.get("status") == "open" and not interview.get("stale"):
            phase = interview.get("phase")
            break
    if session.get("status") == "closed":
        stage = "closed"
    elif active:
        stage = "insight"
    else:
        stage = "intake"
    return {
        "id": session["sessionId"],
        "campaignId": session["campaignId"],
        "stage": stage,
        "phase": phase,
        "openedAt": _stamp(session.get("openedAt")),
        "closedAt": _stamp(session["closedAt"]) if session.get("closedAt") else None,
        "grantId": grant.get("grantId") if active else None,
        "evidenceReady": bool(active and not session.get("insightLocked")),
        "knowledge": _knowledge(session),
    }


def _knowledge(session: dict) -> dict:
    index = session.get("knowledgeIndex") or {}
    warning = index.get("warning")
    return {
        "ready": bool(session.get("knowledgeReady")),
        "vectorCount": int(index.get("vectorCount") or 0),
        "retrieval": index.get("retrieval") or "",
        "vectorStore": index.get("vectorStore") or "session",
        "graphNodes": int(index.get("graphNodes") or 0),
        "graphEdges": int(index.get("graphEdges") or 0),
        "edges": list(index.get("edges") or []),
        "warning": warning if isinstance(warning, str) else None,
        "mcp": _mcp_rows(session, list(index.get("mcp") or []), set()),
    }


def _match_interview(session: dict, interview_id: str, missing_ok: bool = False) -> dict | None:
    found = next(
        (interview for interview in session.get("interviews") or [] if interview.get("interviewId") == interview_id),
        None,
    )
    if found or missing_ok:
        return found
    raise ResearchError("INTERVIEW_MISSING", f"No interview with id {interview_id}.")


def _kinds(session: dict) -> dict[str, str]:
    kinds = {}
    for document in session.get("documents") or []:
        if document.get("id"):
            kinds[document["id"]] = document.get("kind") or "transcript"
    for document in session.get("externalEvidence") or []:
        if document.get("id"):
            kinds[document["id"]] = document.get("kind") or "web"
    return kinds


def _interview(session: dict, interview: dict) -> dict:
    return {
        "id": interview["interviewId"],
        "sessionId": session["sessionId"],
        "cardId": interview.get("cardId") or "",
        "status": "ended" if interview.get("status") == "ended" else "open",
        "phase": interview.get("phase") or "rapport",
        "turnsUsed": interview.get("turnsUsed") or 0,
        "turnCap": max(int(interview.get("turnCap") or 0), TURN_CAP),
        "turns": _turns(session, interview),
    }


def _turns(session: dict, interview: dict) -> list[dict]:
    used: set[str] = set()
    return [_turn(session, turn, None, used) for turn in interview.get("turns") or []]


def _turn(session: dict, turn: dict, follow_up: str | None, used: set[str] | None = None) -> dict:
    kinds = _kinds(session)
    synthesis = _synthesis_for(session, turn)
    payload = {
        "id": turn["id"],
        "phase": turn.get("phase") or "rapport",
        "question": turn.get("question") or "",
        "reply": _reply(turn, kinds, synthesis.get("documentIds") or []),
        "createdAt": _stamp(turn.get("createdAt") or session.get("openedAt")),
        "followUp": follow_up,
        "synthesis": synthesis,
    }
    confidence = turn.get("confidence") or {}
    if confidence:
        payload["confidence"] = {
            "score": float(confidence.get("score") or 0),
            "label": str(confidence.get("label") or ""),
            "basis": str(confidence.get("basis") or ""),
        }
    mcp = _mcp_rows(session, turn.get("mcp") or [], used if used is not None else set())
    if mcp:
        payload["mcp"] = mcp
    return payload


def _mcp_rows(session: dict, calls: list[dict], used: set[str]) -> list[dict]:
    external = list(session.get("externalEvidence") or [])
    rows = []
    for item in calls:
        url = str(item.get("url") or "")
        text = " ".join(str(item.get("text") or "").split())
        if not text:
            text = _mcp_excerpt(item, external, used)
        rows.append(
            {
                "database": str(item.get("database") or ""),
                "tool": str(item.get("tool") or ""),
                "ok": bool(item.get("ok")),
                "error": str(item.get("error") or ""),
                "url": url,
                "text": text[:500],
            }
        )
    return rows


def _mcp_excerpt(item: dict, external: list[dict], used: set[str]) -> str:
    if not item.get("ok") or str(item.get("tool") or "").endswith("upsert"):
        return ""
    url = str(item.get("url") or "")
    tool = str(item.get("tool") or "")
    if url:
        matches = [document for document in external if document.get("url") == url]
        if tool == "search":
            preferred = [document for document in matches if str(document.get("text") or "").startswith("Found ")]
        elif tool == "fetch_content":
            preferred = [document for document in matches if not str(document.get("text") or "").startswith("Found ")]
        else:
            preferred = matches
        chosen = (preferred or matches)[:1]
        if chosen:
            return " ".join(str(chosen[0].get("text") or "").split())
    database = item.get("database") or ""
    for document in external:
        document_id = str(document.get("id") or "")
        if document_id in used or document.get("database") != database:
            continue
        used.add(document_id)
        return " ".join(str(document.get("text") or "").split())
    return ""


def _synthesis_for(session: dict, turn: dict) -> dict:
    if turn.get("abstain"):
        return {"mode": "none", "directDocumentId": None, "conflicts": []}
    if turn.get("source") == "system":
        return {"mode": "system", "directDocumentId": None, "conflicts": []}
    cohort_id = _cohort_for_turn(session, turn)
    documents = [
        document
        for document in session.get("documents") or []
        if not cohort_id or cohort_id in (document.get("cohortIds") or [])
    ]
    documents.extend(session.get("externalEvidence") or [])
    plan = plan_synthesis(
        str(turn.get("question") or ""),
        documents,
        retrieved_ids=set(turn.get("retrievedIds") or []),
    )
    view = _synthesis_view(public_synthesis(plan))
    shown = visible_evidence_ids(str(turn.get("reply") or ""), list(turn.get("evidenceIds") or []))
    view["documentIds"] = shown
    if view.get("directDocumentId") not in shown:
        view["directDocumentId"] = shown[0] if shown else None
    return view


def _synthesis_view(synthesis: dict) -> dict:
    return {
        "mode": synthesis.get("mode") or "none",
        "directDocumentId": synthesis.get("directDocumentId"),
        "documentIds": list(synthesis.get("documentIds") or []),
        "conflicts": [
            {
                "point": conflict.get("point") or "",
                "label": conflict.get("label") or "",
                "sides": [
                    {
                        "stance": side.get("stance") or "",
                        "label": side.get("label") or "",
                        "documentIds": list(side.get("documentIds") or []),
                        "excerpt": side.get("excerpt") or "",
                    }
                    for side in conflict.get("sides") or []
                ],
            }
            for conflict in synthesis.get("conflicts") or []
        ],
    }


def _cohort_for_turn(session: dict, turn: dict) -> str | None:
    turn_id = turn.get("id")
    for interview in session.get("interviews") or []:
        if any(item.get("id") == turn_id for item in interview.get("turns") or []):
            return interview.get("cohortId")
    return None


def _reply(turn: dict, kinds: dict[str, str], document_ids: list[str]) -> dict:
    if turn.get("abstain"):
        return {"kind": "abstain", "reason": turn.get("abstainReason") or "abstain"}
    if turn.get("source") == "system":
        return {
            "kind": "system",
            "text": turn.get("reply") or "No evidence was found. This answer is from the system.",
            "datasets": list(turn.get("systemDatasets") or []),
        }
    chips = []
    seen = set()
    for evidence_id in visible_evidence_ids(str(turn.get("reply") or ""), list(turn.get("evidenceIds") or [])):
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        kind = kinds.get(evidence_id, "transcript")
        chips.append({"spanId": evidence_id, "kind": kind if kind in EVIDENCE_KINDS else "transcript"})
    reply_text = str(turn.get("reply") or "").strip()
    if not chips or not reply_text:
        if reply_text and not turn.get("abstain"):
            basis = str((turn.get("confidence") or {}).get("basis") or "")
            if basis == "uncited" or not list(turn.get("evidenceIds") or []):
                return {"kind": "uncited", "text": turn["reply"], "reason": "uncited"}
        return {"kind": "abstain", "reason": turn.get("abstainReason") or "uncited"}
    return {"kind": "cited", "text": turn["reply"], "chips": chips}


def _pack_view(pack: dict) -> dict:
    return {
        "status": pack.get("status") or "draft",
        "badge": pack.get("badge") or "exploratory",
        "disclosureNote": pack.get("disclosureNote") or "Exploratory. Not a population claim.",
        "oversightName": pack.get("oversightName"),
        "citedTurnCount": int(pack.get("citedTurnCount") or 0),
        "cohorts": [
            {
                "cohortId": cohort.get("cohortId") or "",
                "promise": cohort.get("promise") or "",
                "tension": cohort.get("tension") or "",
                "themes": [
                    {
                        "theme": theme.get("theme") or "",
                        "quote": theme.get("quote") or "",
                        "evidenceIds": list(theme.get("evidenceIds") or []),
                    }
                    for theme in cohort.get("themes") or []
                ],
            }
            for cohort in pack.get("cohorts") or []
        ],
        "skips": [
            {"cohortId": skip.get("cohortId") or "", "reason": skip.get("reason") or ""}
            for skip in pack.get("skips") or []
        ],
    }


class _Handler(BaseHTTPRequestHandler):
    api: ResearchApi

    def do_GET(self) -> None:  # noqa: N802
        match = SESSION_LOG.match(urlparse(self.path).path)
        if match:
            payload = read_session_log(match.group(1))
            if payload is None:
                self._send(
                    {"ok": False, "code": "NOT_FOUND", "message": "No question log for this session yet."},
                    status=404,
                )
                return
            self._send(payload)
            return
        self._dispatch(None)

    def do_PATCH(self) -> None:  # noqa: N802
        self._read_json_body()

    def do_POST(self) -> None:  # noqa: N802
        self._read_json_body()

    def _read_json_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send({"ok": False, "code": "PACK_INVALID", "message": "The request body was not JSON."})
            return
        if not isinstance(body, dict):
            self._send({"ok": False, "code": "PACK_INVALID", "message": "The request body must be an object."})
            return
        self._dispatch(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[research] {self.address_string()} {fmt % args}", flush=True)

    def _dispatch(self, body: dict | None) -> None:
        actor = self.headers.get("X-Actor") or "Meera"
        try:
            payload = self.api.handle(self.command, self.path, body, actor)
        except Exception as error:  # noqa: BLE001
            payload = {"ok": False, "code": "FAULT", "message": str(error)[:300]}
        self._send(payload)

    def _send(self, payload: dict, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve(api: ResearchApi, host: str, port: int) -> None:
    _Handler.api = api
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Research API listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    config = read_config(ROOT)
    enable_session_mcp(ROOT)
    assert_inference_config(config)
    chat = ChatClient(config)
    engine = ResearchEngine(
        SessionStore(ROOT / "data"),
        chat,
        embedder=EmbedClient(config),
        mcp=ResearchMcp(),
    )
    port = int(os.environ.get("PORT") or os.environ.get("RESEARCH_PORT") or 8787)
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    campaign_dir = ROOT / "data" / "campaigns"
    packs = {**load_packs(ROOT / "gold"), **load_packs(campaign_dir)}
    serve(ResearchApi(engine, packs, campaign_dir), host, port)


if __name__ == "__main__":
    main()
