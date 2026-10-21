import json
import os
import re
import uuid
from datetime import datetime, timezone

from synthetic_research.corpus import campaign_sources, corpus_hash, find_pii, validate_campaign
from synthetic_research.gates import (
    ResearchError,
    assert_brand_allowed,
    kit_locked,
    mentions_brand,
    next_phase,
    phase_has_citation,
    public_evidence,
    quote_is_verbatim,
)
from synthetic_research.grounding import grounding_confidence, lock_citations, passage_status
from synthetic_research.knowledge import build_graph, chunk_text, read_upload
from synthetic_research.respondent_turn import (
    bounded_model_evidence,
    enforce_required_statements,
    prior_commitments,
    prior_turns,
    required_statements,
    restore_cited_specifics,
)
from synthetic_research.retrieve import SOURCE_KINDS, tokenize
from synthetic_research.s3_log import log_question
from synthetic_research.search import retrieve_evidence
from synthetic_research.synthesis import blank_synthesis, cited_ids, plan_synthesis, public_synthesis

TURN_CAP = 50


def should_follow_up(turn: dict, turns_left: int) -> bool:
    if turn.get("abstain") or turns_left < 1:
        return False
    return len(str(turn.get("reply") or "").split()) <= 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mcp_points(documents: list[dict]) -> list[dict]:
    points = []
    for document in documents:
        vector = document.get("vector")
        if not document.get("id") or not isinstance(vector, list) or not vector:
            continue
        points.append(
            {
                "id": document["id"],
                "text": str(document.get("text") or "")[:2000],
                "vector": vector,
            }
        )
    return points


def _content_type(filename: str) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return {
        "txt": "text/plain",
        "md": "text/markdown",
        "csv": "text/csv",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf": "application/pdf",
    }.get(suffix, "application/octet-stream")


def _vector_count(index: dict | None) -> int:
    if not index:
        return 0
    return sum(1 for document in index.get("documents") or [] if document.get("vector"))


def _uploaded_vectors(index: dict | None) -> int:
    if not index:
        return 0
    return sum(1 for document in index.get("documents") or [] if document.get("parentId") and document.get("vector"))


def _replace_uploads_with_chunks(session: dict) -> None:
    cohort_ids = [rule["id"] for rule in session.get("cohortRules") or []]
    kept = [document for document in session.get("documents") or [] if not document.get("uploaded")]
    for record in session.get("knowledgeFiles") or []:
        text = record.get("text") or ""
        if not text:
            continue
        for index, chunk in enumerate(chunk_text(text), start=1):
            kept.append(
                {
                    "id": f"{record['id']}-c{index}",
                    "kind": "transcript",
                    "cohortIds": cohort_ids,
                    "title": f"{record.get('filename') or record['id']} · part {index}",
                    "text": chunk,
                    "uploaded": True,
                    "parentId": record["id"],
                }
            )
    session["documents"] = kept


def _merge_graph_evidence(session: dict, question: str, evidence: list[dict]) -> list[dict]:
    graph = session.get("knowledgeGraph") or {}
    tokens = set(tokenize(question))
    if not tokens or not graph.get("edges"):
        return evidence
    matched = set()
    for node in graph.get("nodes") or []:
        if node.get("kind") != "entity":
            continue
        label = set(tokenize(node.get("label") or ""))
        if label and label <= tokens:
            matched.add(node.get("id"))
    if not matched:
        return evidence
    document_ids = {
        edge.get("from")
        for edge in graph.get("edges") or []
        if edge.get("relation") == "mentions" and edge.get("to") in matched
    }
    by_id = {document.get("id"): document for document in session.get("documents") or []}
    seen = {item.get("id") for item in evidence}
    merged = list(evidence)
    for document_id in sorted(item for item in document_ids if item):
        document = by_id.get(document_id)
        if not document or document_id in seen:
            continue
        merged.append(
            {
                "id": document["id"],
                "kind": document.get("kind"),
                "title": document.get("title", ""),
                "text": document.get("text", ""),
                "cohortIds": list(document.get("cohortIds") or []),
                "score": 0.55,
            }
        )
        seen.add(document_id)
    return merged


class ResearchEngine:
    def __init__(self, store, llm, embedder=None, mcp=None):
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.mcp = mcp

    def open(self, campaign: dict, actor: str) -> dict:
        validate_campaign(campaign)
        if not campaign.get("kit", {}).get("locked"):
            raise ResearchError("KIT_UNLOCKED", "The brand kit must be locked before a session can open.")
        session = {
            "sessionId": str(uuid.uuid4()),
            "campaignId": campaign["campaignId"],
            "actor": actor,
            "status": "open",
            "openedAt": _now(),
            "closedAt": None,
            "kit": {
                "version": campaign["kit"]["version"],
                "locked": True,
                "brandAlias": campaign["kit"]["brandAlias"],
                "purpose": campaign["kit"]["purpose"],
                "persona": str(campaign["kit"].get("persona") or "").strip(),
            },
            "marketPack": campaign["marketPack"],
            "cohortRules": [dict(rule) for rule in campaign["cohorts"]],
            "guideTemplate": campaign["guide"],
            "documents": campaign_sources(campaign),
            "grant": None,
            "insightLocked": True,
            "cards": [],
            "guide": None,
            "interviews": [],
            "skips": [],
            "pack": None,
            "packNotes": [],
            "corpusHash": corpus_hash(campaign),
        }
        created = self.store.create(session)
        self._audit(created["sessionId"], "SESSION_OPENED", actor=actor, corpusHash=created["corpusHash"])
        return created

    def grant(self, session_id: str, grant_id: str, steward: str) -> dict:
        session = self._open_session(session_id)
        documents = [doc for doc in session["documents"] if doc["kind"] in SOURCE_KINDS]
        if not documents:
            raise ResearchError("GOLD_EMPTY", "The gold slice has no transcript, review, or screener rows.")
        pii = find_pii(documents)
        if pii:
            raise ResearchError("PII_REJECTED", f"Gold documents contain contact data: {', '.join(pii)}.")
        indexed, retrieval, warning = self._index_documents(documents)
        session["grant"] = {
            "grantId": grant_id,
            "steward": steward,
            "status": "active",
            "documentCount": len(documents),
            "grantedAt": _now(),
            "retrieval": retrieval,
            "corpusHash": session["corpusHash"],
            "embeddingWarning": warning,
        }
        session["insightLocked"] = False
        session["guide"] = _fresh_guide(session["guideTemplate"])
        self.store.write_index(session_id, {"grantId": grant_id, "retrieval": retrieval, "documents": indexed})
        self._audit(
            session_id,
            "GRANT_ACTIVATED",
            steward=steward,
            grantId=grant_id,
            documents=len(documents),
            retrieval=retrieval,
            corpusHash=session["corpusHash"],
        )
        return self.store.write(session)

    def list_knowledge(self, session_id: str) -> list[dict]:
        session = self._open_session(session_id)
        listed = []
        for item in session.get("knowledgeFiles") or []:
            listed.append({key: value for key, value in item.items() if key != "text"})
        return listed

    def add_knowledge(self, session_id: str, filename: str, payload: bytes, actor: str) -> dict:
        session = self._open_session(session_id)
        if session.get("status") == "closed":
            raise ResearchError("SESSION_CLOSED", "This session is closed.")
        text = read_upload(filename, payload)
        if find_pii([{"id": "upload", "title": filename, "text": text}]):
            raise ResearchError("PII_REJECTED", f"{filename} contains contact data and was not added.")
        files = session.setdefault("knowledgeFiles", [])
        file_id = f"UP-{len(files) + 1:03d}"
        cohort_ids = [rule["id"] for rule in session.get("cohortRules") or []]
        session.setdefault("documents", []).append(
            {
                "id": file_id,
                "kind": "transcript",
                "cohortIds": cohort_ids,
                "title": filename,
                "text": text,
                "uploaded": True,
            }
        )
        record = {
            "id": file_id,
            "filename": filename,
            "sizeBytes": len(payload),
            "contentType": _content_type(filename),
            "uploadedAt": _now(),
            "uploadedBy": actor,
            "status": "received",
            "statusLabel": "Received. Vectors and the graph have not been built yet.",
            "text": text,
        }
        files.append(record)
        session["knowledgeReady"] = False
        self._audit(session_id, "KNOWLEDGE_ADDED", fileId=file_id, filename=filename, actor=actor)
        self.store.write(session)
        return record

    def prepare_knowledge(self, session_id: str, steward: str) -> dict:
        session = self._open_session(session_id)
        uploads = session.get("knowledgeFiles") or []
        if not uploads:
            raise ResearchError("KNOWLEDGE_EMPTY", "Upload knowledge before building vectors and a graph.")
        name = str(steward or "").strip() or session.get("actor") or "Meera"
        _replace_uploads_with_chunks(session)
        self.store.write(session)
        if session.get("grant") and session["grant"].get("status") == "active":
            self.refresh(session_id, steward=name)
            session = self.store.read(session_id)
            retrieval = session["grant"].get("retrieval") or "lexical"
            warning = session["grant"].get("embeddingWarning")
            vector_count = _vector_count(self.store.read_index(session_id))
        else:
            granted = self.grant(session_id, f"knowledge-{session_id[:8]}", steward=name)
            retrieval = granted["grant"].get("retrieval") or "lexical"
            warning = granted["grant"].get("embeddingWarning")
            vector_count = _vector_count(self.store.read_index(session_id))
            session = granted
        chunks = [doc for doc in session.get("documents") or [] if doc.get("parentId")]
        graph = build_graph(chunks)
        vector_count = _uploaded_vectors(self.store.read_index(session_id))
        if vector_count < 1:
            raise ResearchError(
                "VECTORS_MISSING",
                warning or "The embedding service did not return a vector for the uploaded knowledge.",
            )
        for record in session.get("knowledgeFiles") or []:
            record["status"] = "ready"
            record["statusLabel"] = "Vectors and the graph include this file."
        session["knowledgeReady"] = True
        session["knowledgeGraph"] = graph
        indexed = (self.store.read_index(session_id) or {}).get("documents") or []
        mcp_rows = self._index_mcp(session, indexed, graph)
        vector_store = "session"
        if any(row.get("ok") and row.get("database") == "vector" for row in mcp_rows):
            vector_store = "mcp"
        session["knowledgeIndex"] = {
            "ready": True,
            "vectorCount": vector_count,
            "vectorStore": vector_store,
            "retrieval": retrieval,
            "graphNodes": len(graph["nodes"]),
            "graphEdges": len(graph["edges"]),
            "edges": graph["edges"][:12],
            "warning": warning,
            "mcp": [_public_mcp(row) for row in mcp_rows],
        }
        self._audit(
            session_id,
            "KNOWLEDGE_PREPARED",
            vectors=vector_count,
            graphNodes=len(graph["nodes"]),
            graphEdges=len(graph["edges"]),
            retrieval=retrieval,
        )
        return self.store.write(session)

    def refuse(self, session_id: str, steward: str, reason: str) -> dict:
        session = self._open_session(session_id)
        session["grant"] = {
            "grantId": None,
            "steward": steward,
            "status": "refused",
            "reason": reason,
            "documentCount": 0,
        }
        session["insightLocked"] = True
        self.store.drop_index(session_id)
        self._audit(session_id, "GRANT_REFUSED", steward=steward)
        return self.store.write(session)

    def revoke(self, session_id: str, steward: str) -> dict:
        session = self._open_session(session_id)
        if not session["grant"] or session["grant"]["status"] != "active":
            raise ResearchError("GRANT_MISSING", "There is no active grant to revoke.")
        session["grant"]["status"] = "revoked"
        session["grant"]["revokedBy"] = steward
        session["grant"]["revokedAt"] = _now()
        session["insightLocked"] = True
        self.store.drop_index(session_id)
        self._audit(session_id, "GRANT_REVOKED", steward=steward)
        return self.store.write(session)

    def list_cohorts(self, session_id: str) -> list[dict]:
        session = self._insight_session(session_id)
        payload = self._index(session)
        rows = []
        for rule in session["cohortRules"]:
            screener = next(
                (
                    doc
                    for doc in payload["documents"]
                    if doc["kind"] == "screener" and rule["id"] in doc["cohortIds"] and doc.get("members")
                ),
                None,
            )
            rows.append(
                {
                    "id": rule["id"],
                    "rule": rule["rule"],
                    "members": screener["members"] if screener else 0,
                    "primaryCard": any(card["cohortId"] == rule["id"] for card in session["cards"]),
                }
            )
        return rows

    def compile_cards(self, session_id: str) -> list[dict]:
        session = self._insight_session(session_id)
        cohorts = self.list_cohorts(session_id)
        if all(cohort["members"] == 0 for cohort in cohorts):
            raise ResearchError("COHORTS_EMPTY", "No cohort in this grant has members.")
        payload = self._index(session)
        cards = []
        for rule in session["cohortRules"]:
            docs = [doc for doc in payload["documents"] if rule["id"] in doc["cohortIds"]]
            members = next(cohort["members"] for cohort in cohorts if cohort["id"] == rule["id"])
            cards.append(
                {
                    "cardId": rule["cardId"],
                    "cohortId": rule["id"],
                    "label": rule["label"],
                    "bounds": list(rule["bounds"]),
                    "traits": list(rule["traits"]),
                    "tension": rule["tension"],
                    "affinity": list(rule["affinity"]),
                    "provenance": {
                        "transcripts": sum(1 for doc in docs if doc["kind"] == "transcript"),
                        "reviews": sum(1 for doc in docs if doc["kind"] == "review"),
                        "sales": sum(1 for doc in docs if doc["kind"] == "sales"),
                        "members": members,
                    },
                }
            )
        session["cards"] = cards
        self.store.write(session)
        return cards

    def edit_screener(self, session_id: str, cohort_id: str, bounds: list[str]) -> dict:
        session = self._insight_session(session_id)
        card = _card(session, cohort_id)
        if not bounds:
            raise ResearchError("BOUNDS_EMPTY", "Screener bounds cannot be empty.")
        card["bounds"] = [str(bound) for bound in bounds]
        card["screenerEdited"] = True
        self.store.write(session)
        return card

    def add_probe(self, session_id: str, text: str) -> dict:
        session = self._insight_session(session_id)
        if not session["guide"]:
            raise ResearchError("GUIDE_MISSING", "The guide is not loaded.")
        probe_text = str(text or "").strip()
        if not probe_text:
            raise ResearchError("PROBE_EMPTY", "A probe needs text.")
        probe = {
            "id": f"probe-{len(session['guide']['probes']) + 1}",
            "text": probe_text,
            "when": "Added this session",
            "sessionOnly": True,
        }
        session["guide"]["probes"].append(probe)
        self.store.write(session)
        return probe

    def edit_main(self) -> None:
        raise ResearchError("MAIN_LOCKED", "Main questions stay locked so cohorts remain comparable.")

    def add_cohort(self) -> None:
        raise ResearchError("COHORT_LOCKED", "Interviews cannot write MDM cohort rules.")

    def start_interview(self, session_id: str, cohort_id: str) -> dict:
        session = self._insight_session(session_id)
        card = _card(session, cohort_id)
        if not session["guide"]:
            raise ResearchError("GUIDE_MISSING", "Load the guide before the interview.")
        existing = _latest_interview(session, cohort_id)
        if existing and existing["status"] == "open":
            return existing
        interview = {
            "interviewId": str(uuid.uuid4()),
            "cohortId": cohort_id,
            "cardId": card["cardId"],
            "status": "open",
            "phase": "rapport",
            "turnCap": TURN_CAP,
            "turnsUsed": 0,
            "turns": [],
            "stale": False,
            "kitLocked": True,
        }
        session["interviews"].append(interview)
        self.store.write(session)
        return interview

    def ask(self, session_id: str, cohort_id: str, question: str) -> dict:
        session = self._insight_session(session_id)
        interview = _open_interview(session, cohort_id)
        if int(interview.get("turnCap") or 0) < TURN_CAP:
            interview["turnCap"] = TURN_CAP
        if interview["turnsUsed"] >= interview["turnCap"]:
            raise ResearchError("TURN_CAP", f"The {TURN_CAP}-turn cap is reached.")
        assert_brand_allowed(interview["phase"], question, session["kit"]["brandAlias"])
        try:
            turn = self._answer_turn(session, interview, question=str(question), instruction="")
        except Exception as error:
            log_question(
                {
                    "at": _now(),
                    "sessionId": session.get("sessionId"),
                    "campaignId": session.get("campaignId"),
                    "interviewId": interview.get("interviewId"),
                    "cohortId": interview.get("cohortId"),
                    "phase": interview.get("phase"),
                    "question": str(question),
                    "error": str(error)[:500],
                    "model": os.environ.get("CHAT_MODEL") or os.environ.get("DEEPINFRA_FAST_MODEL") or "",
                }
            )
            raise
        interview["turns"].append(turn)
        interview["turnsUsed"] += 1
        interview["kitLocked"] = kit_locked(interview["phase"])
        self.store.write(session)
        self._log_question(session, interview, turn)
        return turn

    def follow_up_question(self, session_id: str, cohort_id: str, turn: dict) -> str | None:
        session = self._insight_session(session_id)
        interview = _open_interview(session, cohort_id)
        turns_left = interview["turnCap"] - interview["turnsUsed"]
        if not should_follow_up(turn, turns_left):
            return None
        probes = [probe["text"] for probe in (session.get("guide") or {}).get("probes") or []]
        if not hasattr(self.llm, "follow_up"):
            return probes[0] if probes else None
        raw = self.llm.follow_up(
            json.dumps(
                {
                    "phase": interview["phase"],
                    "brandAllowed": interview["phase"] == "prompted",
                    "brandAlias": session["kit"]["brandAlias"],
                    "probes": probes,
                    "question": turn["question"],
                    "reply": turn["reply"],
                }
            )
        )
        question = str(raw.get("question") or "").strip()
        if not question:
            return None
        if kit_locked(interview["phase"]) and mentions_brand(question, session["kit"]["brandAlias"]):
            return probes[0] if probes else None
        return question

    def improve(self, session_id: str, cohort_id: str, instruction: str) -> dict:
        session = self._insight_session(session_id)
        interview = _open_interview(session, cohort_id)
        current = next((turn for turn in reversed(interview["turns"]) if turn.get("question")), None)
        if not current:
            raise ResearchError("TURN_MISSING", "There is no turn to improve.")
        replacement = self._answer_turn(
            session,
            interview,
            question=current["question"],
            instruction=instruction,
        )
        replacement["id"] = current["id"]
        changed = replacement["reply"] != current["reply"] or replacement["evidenceIds"] != current["evidenceIds"]
        replacement["improved"] = changed
        index = next(i for i, turn in enumerate(interview["turns"]) if turn["id"] == current["id"])
        interview["turns"][index] = replacement
        self.store.write(session)
        self._log_question(session, interview, replacement)
        return replacement

    def _log_question(self, session: dict, interview: dict, turn: dict) -> None:
        try:
            documents = _log_documents(self, session, turn)
            log_question(
                {
                    "at": turn.get("createdAt"),
                    "turnId": turn.get("id"),
                    "sessionId": session.get("sessionId"),
                    "campaignId": session.get("campaignId"),
                    "interviewId": interview.get("interviewId"),
                    "cohortId": interview.get("cohortId"),
                    "phase": turn.get("phase"),
                    "question": turn.get("question"),
                    "reply": turn.get("reply"),
                    "abstain": bool(turn.get("abstain")),
                    "abstainReason": turn.get("abstainReason"),
                    "evidenceIds": list(turn.get("evidenceIds") or []),
                    "documents": documents,
                    "synthesis": turn.get("synthesis") or {},
                    "confidence": turn.get("confidence") or {},
                    "mcp": list(turn.get("mcp") or []),
                    "source": turn.get("source"),
                    "improved": bool(turn.get("improved")),
                    "model": os.environ.get("CHAT_MODEL") or os.environ.get("DEEPINFRA_FAST_MODEL") or "",
                }
            )
        except Exception as error:  # noqa: BLE001
            print(f"[s3-log] {type(error).__name__}: {error}", flush=True)

    def open_evidence(self, session_id: str, evidence_id: str) -> dict:
        session = self._insight_session(session_id)
        payload = self._index(session)
        document = next((doc for doc in payload["documents"] if doc["id"] == evidence_id), None)
        if not document:
            document = next(
                (doc for doc in session.get("externalEvidence") or [] if doc["id"] == evidence_id),
                None,
            )
        if not document:
            raise ResearchError("EVIDENCE_MISSING", f"Evidence {evidence_id} is not in this grant.")
        return public_evidence(document, session["grant"]["grantId"])

    def advance_phase(self, session_id: str, cohort_id: str) -> dict:
        session = self._insight_session(session_id)
        interview = _open_interview(session, cohort_id)
        if not phase_has_citation(interview, interview["phase"]):
            raise ResearchError("PHASE_BLOCKED", "Advance only after this phase has a cited reply.")
        upcoming = next_phase(interview["phase"])
        if not upcoming:
            raise ResearchError("PHASE_DONE", "The interview is already in Prompted.")
        interview["phase"] = upcoming
        interview["kitLocked"] = kit_locked(upcoming)
        self.store.write(session)
        return interview

    def ladder(self, session_id: str, cohort_id: str) -> dict:
        session = self._insight_session(session_id)
        interview = _open_interview(session, cohort_id)
        if interview["phase"] not in {"deep_dive", "prompted"}:
            raise ResearchError("LADDER_PHASE", "Laddering belongs in the deep dive or Prompted.")
        probes = session["guide"]["probes"]
        if not probes:
            raise ResearchError("PROBE_MISSING", "The probe bank is empty.")
        previous = next((turn["reply"] for turn in reversed(interview["turns"]) if turn.get("reply")), "")
        question = probes[0]["text"] if not previous else f"{probes[0]['text']} Last reply: {previous}"
        return self.ask(session_id, cohort_id, question)

    def end_interview(self, session_id: str, cohort_id: str) -> dict:
        session = self._insight_session(session_id)
        interview = _open_interview(session, cohort_id)
        if not any(not turn["abstain"] and turn["evidenceIds"] for turn in interview["turns"]):
            raise ResearchError("INTERVIEW_EMPTY", "End only after at least one cited reply.")
        interview["status"] = "ended"
        interview["endedAt"] = _now()
        self.store.write(session)
        return interview

    def skip(self, session_id: str, cohort_id: str, reason: str) -> list[dict]:
        session = self._insight_session(session_id)
        text = str(reason or "").strip()
        if not text:
            raise ResearchError("SKIP_REASON", "A skip needs a written reason.")
        interview = _latest_interview(session, cohort_id)
        if interview and interview["status"] == "ended":
            raise ResearchError("SKIP_CLOSED", "An ended interview cannot be skipped.")
        session["skips"] = [item for item in session["skips"] if item["cohortId"] != cohort_id]
        session["skips"].append({"cohortId": cohort_id, "reason": text, "at": _now()})
        self.store.write(session)
        return session["skips"]

    def draft_pack(self, session_id: str) -> dict:
        session = self._insight_session(session_id)
        usable = [
            interview
            for interview in session["interviews"]
            if interview["status"] == "ended" and not interview["stale"]
        ]
        if not usable:
            raise ResearchError(
                "PACK_BLOCKED",
                "Draft a pack only after at least one interview has ended and is not stale.",
            )
        cited = []
        for interview in usable:
            for turn in interview["turns"]:
                if turn["abstain"] or not turn["evidenceIds"]:
                    continue
                synthesis = turn.get("synthesis") or {}
                cited.append(
                    {
                        "cohortId": interview["cohortId"],
                        "phase": turn["phase"],
                        "question": turn["question"],
                        "reply": turn["reply"],
                        "evidenceIds": turn["evidenceIds"],
                        "directDocumentId": synthesis.get("directDocumentId"),
                        "conflicts": synthesis.get("conflicts") or [],
                    }
                )
        prompt = json.dumps(
            {
                "campaignId": session["campaignId"],
                "brandAlias": session["kit"]["brandAlias"],
                "notes": session["packNotes"],
                "skips": session["skips"],
                "cards": [
                    {
                        "cohortId": card["cohortId"],
                        "label": card["label"],
                        "tension": card["tension"],
                        "bounds": card["bounds"],
                    }
                    for card in session["cards"]
                ],
                "citedTurns": cited,
            },
            indent=2,
        )
        raw = self.llm.pack(prompt)
        replies_by_cohort: dict[str, list[str]] = {}
        for turn in cited:
            replies_by_cohort.setdefault(turn["cohortId"], []).append(turn["reply"])
        cohorts = []
        for cohort in raw.get("cohorts") or []:
            replies = replies_by_cohort.get(cohort.get("cohortId"), [])
            themes = [theme for theme in cohort.get("themes") or [] if quote_is_verbatim(theme.get("quote", ""), replies)]
            if not themes:
                continue
            cohorts.append(
                {
                    "cohortId": cohort.get("cohortId"),
                    "promise": str(cohort.get("promise") or "").strip(),
                    "tension": str(cohort.get("tension") or "").strip(),
                    "themes": themes,
                }
            )
        if not cohorts:
            raw = self.llm.pack(
                "The previous themes were rejected because each quote must be an exact substring of a reply. "
                "Copy the reply words exactly.\n"
                + prompt
            )
            cohorts = []
            for cohort in raw.get("cohorts") or []:
                replies = replies_by_cohort.get(cohort.get("cohortId"), [])
                themes = [
                    theme
                    for theme in cohort.get("themes") or []
                    if quote_is_verbatim(theme.get("quote", ""), replies)
                ]
                if not themes:
                    continue
                cohorts.append(
                    {
                        "cohortId": cohort.get("cohortId"),
                        "promise": str(cohort.get("promise") or "").strip(),
                        "tension": str(cohort.get("tension") or "").strip(),
                        "themes": themes,
                    }
                )
        session["pack"] = {
            "status": "draft",
            "badge": "exploratory",
            "clientTold": True,
            "oversightName": None,
            "populationClaim": False,
            "disclosureNote": str(raw.get("disclosureNote") or "Exploratory. Not a population claim."),
            "cohorts": cohorts,
            "skips": [dict(skip) for skip in session["skips"]],
            "citedTurnCount": len(cited),
            "draftedAt": _now(),
        }
        self.store.write(session)
        return session["pack"]

    def send_changes(self, session_id: str, notes: str) -> dict:
        session = self._insight_session(session_id)
        if not session["pack"]:
            raise ResearchError("PACK_MISSING", "There is no draft pack to send back.")
        text = str(notes or "").strip()
        if not text:
            raise ResearchError("NOTES_EMPTY", "Send-back needs notes.")
        session["packNotes"].append({"notes": text, "at": _now()})
        session["pack"]["status"] = "changes_requested"
        self.store.write(session)
        return session["pack"]

    def refresh(self, session_id: str, steward: str) -> dict:
        session = self._open_session(session_id)
        if not session["grant"] or session["grant"]["status"] != "active":
            raise ResearchError("GRANT_MISSING", "Refresh needs an active grant.")
        documents = [doc for doc in session["documents"] if doc["kind"] in SOURCE_KINDS]
        indexed, retrieval, warning = self._index_documents(documents)
        session["grant"]["retrieval"] = retrieval
        session["grant"]["embeddingWarning"] = warning
        self.store.write_index(
            session_id,
            {
                "grantId": session["grant"]["grantId"],
                "retrieval": retrieval,
                "documents": indexed,
                "refreshedBy": steward,
                "refreshedAt": _now(),
            },
        )
        for interview in session["interviews"]:
            if interview["status"] == "ended":
                interview["stale"] = True
        session["cards"] = []
        self.store.write(session)
        self.compile_cards(session_id)
        refreshed = self.store.read(session_id)
        refreshed["insightLocked"] = False
        if refreshed["pack"] and refreshed["pack"]["status"] != "approved":
            refreshed["pack"]["status"] = "stale"
        self._audit(session_id, "GRANT_REFRESHED", steward=steward, retrieval=retrieval)
        return self.store.write(refreshed)

    def approve(self, session_id: str, oversight_name: str) -> dict:
        session = self._insight_session(session_id)
        if not session["pack"] or session["pack"]["status"] == "stale":
            raise ResearchError("PACK_MISSING", "Approve needs a current draft pack.")
        name = str(oversight_name or "").strip()
        if not name:
            raise ResearchError("OVERSIGHT_MISSING", "Name the human oversight before approve.")
        if session["pack"]["badge"] != "exploratory":
            raise ResearchError("BADGE_MISSING", "The exploratory badge is required.")
        uncovered = []
        for card in session["cards"]:
            ended = any(
                interview["cohortId"] == card["cohortId"] and interview["status"] == "ended" and not interview["stale"]
                for interview in session["interviews"]
            )
            skipped = any(skip["cohortId"] == card["cohortId"] for skip in session["skips"])
            if not ended and not skipped:
                uncovered.append(card["cohortId"])
        if uncovered:
            raise ResearchError("SKIP_MISSING", "Every cohort must be interviewed or skipped with a reason.")
        session["pack"]["status"] = "approved"
        session["pack"]["oversightName"] = name
        session["pack"]["approvedAt"] = _now()
        vault_pack = {
            "campaignId": session["campaignId"],
            "sessionId": session["sessionId"],
            "grantId": session["grant"]["grantId"],
            "badge": session["pack"]["badge"],
            "clientTold": True,
            "oversightName": name,
            "populationClaim": False,
            "disclosureNote": session["pack"]["disclosureNote"],
            "cohorts": session["pack"]["cohorts"],
            "skips": session["pack"]["skips"],
            "approvedAt": session["pack"]["approvedAt"],
        }
        self.store.promote_pack(session_id, vault_pack)
        self.store.write(session)
        self._audit(session_id, "PACK_APPROVED", oversight=name, corpusHash=session["corpusHash"])
        return vault_pack

    def close(self, session_id: str) -> dict:
        session = self.store.read(session_id)
        session["status"] = "closed"
        session["closedAt"] = _now()
        session["insightLocked"] = True
        self.store.drop_index(session_id)
        self._audit(session_id, "SESSION_CLOSED")
        return self.store.write(session)

    def snapshot(self, session_id: str) -> dict:
        return {
            "session": self.store.read(session_id),
            "indexPresent": self.store.index_exists(session_id),
            "vault": self.store.read_vault(session_id),
            "audit": self.store.read_audit(session_id),
        }

    def _answer_turn(self, session: dict, interview: dict, *, question: str, instruction: str) -> dict:
        card = _card(session, interview["cohortId"])
        outside = _outside_bounds(question, card["bounds"])
        evidence: list[dict] = []
        mcp_results: list[dict] = []
        plan = plan_synthesis(question, [])
        if not outside:
            payload = self._index(session)
            query_vector = None
            if self.embedder and any(document.get("vector") for document in payload["documents"]):
                query_vector = self.embedder.embed([question])[0]
            evidence = retrieve_evidence(
                payload["documents"],
                question,
                cohort_id=interview["cohortId"],
                query_vector=query_vector,
                limit=None,
            )
            evidence = [item for item in evidence if item.get("kind") in {"transcript", "review", "sales"}]
            evidence = _merge_graph_evidence(session, question, evidence)
            group = _synthesis_group(payload["documents"], interview["cohortId"], evidence)
            plan = plan_synthesis(question, group, retrieved_ids={item["id"] for item in evidence})
            transcripts_answer = plan["mode"] in {"direct", "conflict"} and bool(plan.get("reply"))
            if self.mcp is not None:
                if hasattr(self.mcp, "stored") and not self.mcp.stored(session["sessionId"]):
                    mcp_rows = self._index_mcp(session, payload["documents"], session.get("knowledgeGraph") or {})
                    if mcp_rows:
                        index = session.setdefault("knowledgeIndex", {})
                        index["mcp"] = [_public_mcp(row) for row in mcp_rows]
                        if any(row.get("ok") and row.get("database") == "vector" for row in mcp_rows):
                            index["vectorStore"] = "mcp"
                        self.store.write(session)
                if not transcripts_answer:
                    mcp_results = list(
                        self.mcp.gather(
                            question,
                            interview["cohortId"],
                            session_id=session["sessionId"],
                            query_vector=query_vector,
                        )
                        or []
                    )
                    external = [
                        item
                        for item in _external_evidence(session, mcp_results)
                        if item.get("database") == "web"
                    ]
                    if external:
                        session.setdefault("externalEvidence", []).extend(external)
                        evidence = [*evidence, *external]
                        group = _synthesis_group(payload["documents"], interview["cohortId"], evidence)
                        plan = plan_synthesis(question, group, retrieved_ids={item["id"] for item in evidence})
        if outside:
            return _empty_turn(interview, question, "out_of_screener_bounds", mcp_results)
        if plan.get("conflicts") or (plan["mode"] in {"direct", "conflict"} and plan.get("reply")):
            evidence = _ordered_evidence(evidence, group, _evidence_plan(plan))
        if not evidence:
            return self._answer_from_system(session, interview, card, question, instruction, mcp_results)
        return self._answer_from_model(
            session,
            interview,
            card,
            question,
            instruction,
            evidence,
            mcp_results,
            plan,
        )

    def _answer_from_model(
        self,
        session: dict,
        interview: dict,
        card: dict,
        question: str,
        instruction: str,
        evidence: list[dict],
        mcp_results: list[dict],
        plan: dict,
    ) -> dict:
        statements = required_statements(plan)
        model_evidence = bounded_model_evidence(
            [item for item in evidence if not str(item.get("id") or "").startswith("GRAPH-")],
            plan,
        )
        raw = self.llm.respondent(
            json.dumps(
                {
                    "phase": interview["phase"],
                    "brandAlias": session["kit"]["brandAlias"],
                    "brandAllowed": interview["phase"] == "prompted",
                    "bounds": card["bounds"],
                    "tension": card["tension"],
                    "marketPack": session["marketPack"],
                    "persona": _session_persona(session),
                    "question": question,
                    "instruction": instruction,
                    "task": plan.get("mode") or "indirect",
                    "requiredStatements": statements,
                    "priorTurns": prior_turns(interview.get("turns") or []),
                    "priorCommitments": prior_commitments(question, interview.get("turns") or []),
                    "synthesis": public_synthesis(plan),
                    "evidence": [_model_evidence(item) for item in model_evidence],
                },
                indent=2,
            )
        )
        allowed = {item["id"] for item in model_evidence if item.get("id")}
        evidence_ids = [item for item in raw.get("evidenceIds") or [] if item in allowed]
        reply = str(raw.get("reply") or "").strip()
        reply, evidence_ids, removed = lock_citations(reply, evidence_ids, allowed)
        reply, evidence_ids = enforce_required_statements(reply, evidence_ids, statements, allowed)
        evidence_ids = [item for item in evidence_ids if item in allowed]
        if not reply or not evidence_ids:
            return _finish_turn(
                interview,
                question,
                abstain=True,
                reason="uncited",
                reply="",
                evidence_ids=[],
                evidence=evidence,
                mcp_results=mcp_results,
                source=None,
                synthesis=blank_synthesis("none"),
            )
        evidence_ids = [item for item in cited_ids(plan, evidence_ids) if item in allowed]
        prior_document_ids: set[str] = set()
        for prior_turn in interview.get("turns") or []:
            for document_id in prior_turn.get("evidenceIds") or []:
                if document_id:
                    prior_document_ids.add(str(document_id))
        indexed = {doc["id"]: doc for doc in self._index(session)["documents"] if doc.get("id")}
        corpus_for_restore = list(evidence)
        seen_corpus = {str(item.get("id") or "") for item in corpus_for_restore}
        for document_id in prior_document_ids:
            if document_id in indexed and document_id not in seen_corpus:
                corpus_for_restore.append(indexed[document_id])
                seen_corpus.add(document_id)
        retrieval_ids = {str(item["id"]) for item in evidence if item.get("id")} | prior_document_ids
        reply, evidence_ids = restore_cited_specifics(
            reply,
            evidence_ids,
            model_evidence,
            question=question,
            corpus_evidence=corpus_for_restore,
            allowed=allowed,
            retrieval_ids=retrieval_ids,
            prior_document_ids=prior_document_ids,
        )
        evidence_ids = [item for item in evidence_ids if item in retrieval_ids]
        leaked = mentions_brand(reply, session["kit"]["brandAlias"]) and interview["phase"] != "prompted"
        pii = find_pii([{"id": "reply", "title": "", "text": reply}])
        if leaked or pii:
            return _empty_turn(interview, question, "brand_locked" if leaked else "pii", mcp_results)
        synthesis = public_synthesis(plan)
        if plan.get("mode") == "indirect":
            synthesis["mode"] = "indirect"
        turn = _finish_turn(
            interview,
            question,
            abstain=False,
            reason=None,
            reply=reply,
            evidence_ids=evidence_ids,
            evidence=evidence,
            mcp_results=mcp_results,
            source=None,
            synthesis=synthesis,
        )
        if removed:
            used_web = any(str(item).startswith("WEB-") for item in evidence_ids)
            turn["confidence"] = grounding_confidence(supported=True, removed=removed, used_web=used_web)
            turn["grounding"] = {"supported": True, "removed": removed}
        if plan.get("mode") in {"direct", "conflict"}:
            _note_direct_basis(turn, plan)
        return turn

    def _answer_from_system(
        self,
        session: dict,
        interview: dict,
        card: dict,
        question: str,
        instruction: str,
        mcp_results: list[dict],
    ) -> dict:
        datasets = _system_datasets(session, interview["cohortId"])
        raw = self.llm.respondent(
            json.dumps(
                {
                    "source": "system",
                    "phase": interview["phase"],
                    "brandAlias": session["kit"]["brandAlias"],
                    "brandAllowed": interview["phase"] == "prompted",
                    "bounds": card["bounds"],
                    "tension": card["tension"],
                    "marketPack": session["marketPack"],
                    "persona": _session_persona(session),
                    "question": question,
                    "instruction": instruction,
                    "priorTurns": prior_turns(interview.get("turns") or []),
                    "priorCommitments": prior_commitments(question, interview.get("turns") or []),
                    "evidence": [],
                    "systemDatasets": datasets,
                },
                indent=2,
            )
        )
        reply = _system_disclosure(str(raw.get("reply") or "").strip())
        leaked = mentions_brand(reply, session["kit"]["brandAlias"]) and interview["phase"] != "prompted"
        pii = find_pii([{"id": "reply", "title": "", "text": reply}])
        if leaked or pii:
            return _empty_turn(interview, question, "brand_locked" if leaked else "pii", mcp_results)
        turn = _finish_turn(
            interview,
            question,
            abstain=False,
            reason=None,
            reply=reply,
            evidence_ids=[],
            evidence=[],
            mcp_results=mcp_results,
            source="system",
        )
        turn["systemDatasets"] = sorted({row["datasetId"] for row in datasets if row.get("datasetId")})
        return turn

    def _open_session(self, session_id: str) -> dict:
        session = self.store.read(session_id)
        if session["status"] == "closed":
            raise ResearchError("SESSION_CLOSED", "This session is closed.")
        return session

    def _insight_session(self, session_id: str) -> dict:
        session = self._open_session(session_id)
        grant = session.get("grant") or {}
        if grant.get("status") != "active" or session.get("insightLocked"):
            raise ResearchError("INSIGHT_LOCKED", "Insight is locked until a gold grant is active.")
        return session

    def _index(self, session: dict) -> dict:
        payload = self.store.read_index(session["sessionId"])
        if not payload:
            raise ResearchError("INDEX_MISSING", "This session has no index.")
        return payload

    def _audit(self, session_id: str, action: str, **fields) -> None:
        self.store.append_audit(session_id, {"at": _now(), "action": action, **fields})

    def _index_documents(self, documents: list[dict]) -> tuple[list[dict], str, str | None]:
        vectors = None
        warning = None
        retrieval = "lexical"
        if self.embedder is not None:
            try:
                vectors = self.embedder.embed(
                    [f"{document.get('title', '')}\n{document.get('text', '')}" for document in documents]
                )
                retrieval = "hybrid"
            except Exception as error:
                warning = str(error)[:180]
        indexed = []
        for index, document in enumerate(documents):
            copy = {key: value for key, value in document.items() if key != "vector"}
            if vectors is not None:
                copy["vector"] = vectors[index]
            indexed.append(copy)
        return indexed, retrieval, warning

    def _index_mcp(self, session: dict, documents: list[dict], graph: dict) -> list[dict]:
        if self.mcp is None or not hasattr(self.mcp, "index_knowledge"):
            return []
        try:
            return list(self.mcp.index_knowledge(session["sessionId"], _mcp_points(documents), graph or {}) or [])
        except Exception as error:
            return [
                {
                    "database": "vector",
                    "server": "session-knowledge",
                    "tool": "vector_upsert",
                    "ok": False,
                    "text": "",
                    "url": "",
                    "error": str(error)[:180],
                }
            ]


def _synthesis_group(documents: list[dict], cohort_id: str, evidence: list[dict]) -> list[dict]:
    grouped = []
    seen = set()
    for document in documents:
        if document.get("kind") not in SOURCE_KINDS:
            continue
        if cohort_id not in (document.get("cohortIds") or []):
            continue
        if not document.get("id") or document["id"] in seen:
            continue
        grouped.append(document)
        seen.add(document["id"])
    for item in evidence:
        if item.get("id") and item["id"] not in seen:
            grouped.append(item)
            seen.add(item["id"])
    return grouped


def _evidence_plan(plan: dict) -> dict:
    evidence_ids = []
    for document_id in [*(plan.get("evidenceIds") or []), *(plan.get("fallbackEvidenceIds") or [])]:
        if document_id and document_id not in evidence_ids:
            evidence_ids.append(document_id)
    for conflict in plan.get("conflicts") or []:
        for side in conflict.get("sides") or []:
            for document_id in side.get("documentIds") or []:
                if document_id not in evidence_ids:
                    evidence_ids.append(document_id)
    return {**plan, "evidenceIds": evidence_ids}


def _ordered_evidence(retrieved: list[dict], group: list[dict], plan: dict) -> list[dict]:
    by_id: dict[str, dict] = {}
    for document in [*retrieved, *group]:
        document_id = document.get("id")
        if not document_id or document_id in by_id:
            continue
        by_id[document_id] = {
            "id": document_id,
            "kind": document.get("kind"),
            "title": document.get("title", ""),
            "text": document.get("text", ""),
            "cohortIds": list(document.get("cohortIds") or []),
            "score": float(document.get("score") or 0),
            "database": document.get("database"),
            "url": document.get("url") or "",
        }
    for row in plan.get("ranked") or []:
        document_id = row.get("id")
        if document_id not in by_id or row.get("relation") != "direct":
            continue
        by_id[document_id]["score"] = max(by_id[document_id]["score"], min(1.0, float(row.get("score") or 0)))
    if plan.get("mode") in {"direct", "conflict"}:
        for document_id in plan.get("evidenceIds") or []:
            if document_id in by_id:
                by_id[document_id]["score"] = max(by_id[document_id]["score"], 0.55)
    elif plan.get("conflicts"):
        for document_id in plan.get("evidenceIds") or []:
            if document_id in by_id and by_id[document_id]["score"] <= 0:
                by_id[document_id]["score"] = 0.35
    ordered = []
    seen = set()
    for document_id in plan.get("evidenceIds") or []:
        if document_id in by_id and document_id not in seen:
            ordered.append(by_id[document_id])
            seen.add(document_id)
    for item in retrieved:
        document_id = item.get("id")
        if document_id in by_id and document_id not in seen:
            ordered.append(by_id[document_id])
            seen.add(document_id)
    return ordered


def _note_direct_basis(turn: dict, plan: dict) -> None:
    confidence = dict(turn.get("confidence") or {})
    if plan.get("mode") == "direct" and plan.get("directDocumentId"):
        confidence["basis"] = f"direct answer in {plan['directDocumentId']}"
    elif plan.get("mode") == "conflict":
        confidence["basis"] = "direct answers in conflict"
    turn["confidence"] = confidence


def _empty_turn(interview: dict, question: str, reason: str, mcp_results: list[dict] | None = None) -> dict:
    return _finish_turn(
        interview,
        question,
        abstain=True,
        reason=reason,
        reply="",
        evidence_ids=[],
        evidence=[],
        mcp_results=mcp_results or [],
        source=None,
    )


def _model_evidence(item: dict) -> dict:
    row = {
        "id": item["id"],
        "kind": item.get("kind") or "",
        "text": str(item.get("text") or "")[:1200],
    }
    status = passage_status(row["text"])
    if status:
        row["status"] = status
    return row


def _log_documents(engine: ResearchEngine, session: dict, turn: dict) -> list[dict]:
    by_id: dict[str, dict] = {}
    for document in session.get("documents") or []:
        if document.get("id"):
            by_id[document["id"]] = document
    for document in session.get("externalEvidence") or []:
        if document.get("id"):
            by_id[document["id"]] = document
    indexed = engine.store.read_index(session["sessionId"]) or {}
    for document in indexed.get("documents") or []:
        if document.get("id") and document["id"] not in by_id:
            by_id[document["id"]] = document
    cited = set(turn.get("evidenceIds") or [])
    ordered = list(turn.get("retrievedIds") or [])
    for document_id in cited:
        if document_id not in ordered:
            ordered.append(document_id)
    rows = []
    for document_id in ordered:
        document = by_id.get(document_id) or {}
        rows.append(
            {
                "id": document_id,
                "kind": document.get("kind") or "",
                "title": document.get("title") or "",
                "cited": document_id in cited,
                "text": document.get("text") or "",
            }
        )
    return rows


def _finish_turn(
    interview: dict,
    question: str,
    *,
    abstain: bool,
    reason: str | None,
    reply: str,
    evidence_ids: list[str],
    evidence: list[dict],
    mcp_results: list[dict],
    source: str | None,
    synthesis: dict | None = None,
) -> dict:
    if synthesis is not None:
        recorded = synthesis
    elif source == "system":
        recorded = blank_synthesis("system")
    elif abstain:
        recorded = blank_synthesis("none")
    else:
        recorded = blank_synthesis("indirect")
    turn = {
        "id": str(uuid.uuid4()),
        "phase": interview["phase"],
        "question": question,
        "abstain": abstain,
        "abstainReason": reason,
        "reply": reply,
        "evidenceIds": evidence_ids,
        "retrievedIds": [item["id"] for item in evidence],
        "mcp": [_public_mcp(item) for item in mcp_results],
        "confidence": _confidence(evidence, evidence_ids, source),
        "synthesis": recorded,
        "createdAt": _now(),
        "improved": False,
    }
    if source:
        turn["source"] = source
    return turn


def _public_mcp(item: dict) -> dict:
    return {
        "database": item.get("database") or "",
        "server": item.get("server") or "",
        "tool": item.get("tool") or "",
        "ok": bool(item.get("ok")),
        "url": item.get("url") or "",
        "error": item.get("error") or "",
        "text": " ".join(str(item.get("text") or "").split())[:500],
    }


def _confidence(evidence: list[dict], evidence_ids: list[str], source: str | None) -> dict:
    if source == "system":
        return {"score": 0.35, "label": "low", "basis": "system dataset"}
    cited = [item for item in evidence if item.get("id") in evidence_ids]
    if not cited:
        return {"score": 0.0, "label": "none", "basis": "uncited"}
    databases = sorted({item.get("database") for item in cited if item.get("database")})
    basis = "cited score"
    if databases:
        basis = "cited score from " + ", ".join(databases)
    return {"score": 1.0, "label": "high", "basis": basis}


_WEB_WALL = (
    "skip to main content",
    "sign in to view",
    "sign in to continue",
    "join now",
    "cookie policy",
    "open the app",
)


def _web_text_is_usable(text: str) -> bool:
    lowered = text.lower()
    walls = sum(phrase in lowered for phrase in _WEB_WALL)
    return walls < 2


def _external_evidence(session: dict, results: list[dict]) -> list[dict]:
    start = len(session.get("externalEvidence") or []) + 1
    documents = []
    for result in results:
        text = " ".join(str(result.get("text") or "").split())[:1200]
        if not result.get("ok") or not text:
            continue
        if find_pii([{"id": "external", "title": "", "text": text}]):
            result["ok"] = False
            result["error"] = "pii"
            result["text"] = ""
            continue
        if (result.get("database") or "web") == "web" and not _web_text_is_usable(text):
            result["ok"] = False
            result["error"] = "The page had no answer, only navigation or a sign-in wall."
            result["text"] = ""
            continue
        database = result.get("database") or "web"
        prefix = {"vector": "VEC", "graph": "GRAPH"}.get(database, "WEB")
        documents.append(
            {
                "id": f"{prefix}-{start + len(documents)}",
                "kind": "web" if prefix == "WEB" else "transcript",
                "title": result.get("url") or result.get("tool") or database,
                "text": text,
                "cohortIds": [],
                "score": 1.0,
                "database": database,
                "url": result.get("url") or "",
            }
        )
    return documents


def _system_datasets(session: dict, cohort_id: str) -> list[dict]:
    rows = []
    for document in session.get("documents") or []:
        if document.get("kind") != "sales" and not document.get("datasetId"):
            continue
        if cohort_id not in (document.get("cohortIds") or []):
            continue
        rows.append(
            {
                "id": document.get("id"),
                "datasetId": document.get("datasetId"),
                "area": document.get("area") or "",
                "title": document.get("title") or "",
                "text": document.get("text") or "",
            }
        )
    return rows


def _system_disclosure(reply: str) -> str:
    disclosure = "No evidence was found. This answer is from the system."
    text = " ".join((reply or "").split())
    lowered = text.lower()
    if lowered.startswith("no evidence was found. this answer is from the system"):
        return text
    if lowered.startswith("no evidence was found"):
        period = text.find(". ")
        text = text[period + 2:].strip() if period != -1 else ""
    body = text or "I am answering from the system datasets."
    return f"{disclosure} {body}"


def _session_persona(session: dict) -> str:
    kit = session.get("kit") or {}
    return str(kit.get("persona") or kit.get("purpose") or "").strip()


def _fresh_guide(template: dict) -> dict:
    return {
        "mains": [{**main, "locked": True} for main in template["mains"]],
        "probes": [dict(probe) for probe in template["probes"]],
    }


def _card(session: dict, cohort_id: str) -> dict:
    card = next((item for item in session["cards"] if item["cohortId"] == cohort_id), None)
    if not card:
        raise ResearchError("CARD_MISSING", "Compile a card before this step.")
    return card


def _latest_interview(session: dict, cohort_id: str) -> dict | None:
    matches = [
        interview
        for interview in session["interviews"]
        if interview["cohortId"] == cohort_id and not interview["stale"]
    ]
    return matches[-1] if matches else None


def _open_interview(session: dict, cohort_id: str) -> dict:
    interview = _latest_interview(session, cohort_id)
    if not interview or interview["status"] != "open":
        raise ResearchError("INTERVIEW_MISSING", "Start the interview before asking.")
    return interview


def _outside_bounds(question: str, bounds: list[str]) -> bool:
    text = (question or "").lower()
    tea_only_out = any("not tea-only" in bound.lower() for bound in bounds)
    return bool(tea_only_out and re_tea(text))


def re_tea(text: str) -> bool:
    return re.search(r"\btea\b", text) is not None and re.search(r"\bcoffee\b", text) is None
