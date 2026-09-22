import json
import re
import uuid
from datetime import datetime, timezone

from synthetic_research.corpus import corpus_hash, find_pii, validate_campaign
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
from synthetic_research.retrieve import SOURCE_KINDS
from synthetic_research.search import retrieve_evidence

TURN_CAP = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchEngine:
    def __init__(self, store, llm, embedder=None):
        self.store = store
        self.llm = llm
        self.embedder = embedder

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
            },
            "marketPack": campaign["marketPack"],
            "cohortRules": [dict(rule) for rule in campaign["cohorts"]],
            "guideTemplate": campaign["guide"],
            "documents": campaign["documents"],
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
        if interview["turnsUsed"] >= interview["turnCap"]:
            raise ResearchError("TURN_CAP", "The ten-turn cap is reached.")
        assert_brand_allowed(interview["phase"], question, session["kit"]["brandAlias"])
        turn = self._answer_turn(session, interview, question=str(question), instruction="")
        interview["turns"].append(turn)
        interview["turnsUsed"] += 1
        interview["kitLocked"] = kit_locked(interview["phase"])
        self.store.write(session)
        return turn

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
        replacement["improved"] = True
        index = next(i for i, turn in enumerate(interview["turns"]) if turn["id"] == current["id"])
        interview["turns"][index] = replacement
        self.store.write(session)
        return replacement

    def open_evidence(self, session_id: str, evidence_id: str) -> dict:
        session = self._insight_session(session_id)
        payload = self._index(session)
        document = next((doc for doc in payload["documents"] if doc["id"] == evidence_id), None)
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
                cited.append(
                    {
                        "cohortId": interview["cohortId"],
                        "phase": turn["phase"],
                        "question": turn["question"],
                        "reply": turn["reply"],
                        "evidenceIds": turn["evidenceIds"],
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
        evidence = []
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
            )
        if not evidence:
            return {
                "id": str(uuid.uuid4()),
                "phase": interview["phase"],
                "question": question,
                "abstain": True,
                "abstainReason": "out_of_screener_bounds" if outside else "no_evidence",
                "reply": "",
                "evidenceIds": [],
                "retrievedIds": [],
                "improved": False,
            }
        raw = self.llm.respondent(
            json.dumps(
                {
                    "phase": interview["phase"],
                    "brandAlias": session["kit"]["brandAlias"],
                    "brandAllowed": interview["phase"] == "prompted",
                    "bounds": card["bounds"],
                    "tension": card["tension"],
                    "marketPack": session["marketPack"],
                    "question": question,
                    "instruction": instruction,
                    "evidence": [
                        {"id": item["id"], "kind": item["kind"], "text": item["text"]} for item in evidence
                    ],
                },
                indent=2,
            )
        )
        allowed = {item["id"] for item in evidence}
        evidence_ids = [item for item in raw.get("evidenceIds") or [] if item in allowed]
        reply = str(raw.get("reply") or "").strip()
        leaked = mentions_brand(reply, session["kit"]["brandAlias"]) and interview["phase"] != "prompted"
        pii = find_pii([{"id": "reply", "title": "", "text": reply}])
        abstain = bool(raw.get("abstain")) or not evidence_ids or not reply or leaked or bool(pii)
        reason = None
        if abstain:
            if leaked:
                reason = "brand_locked"
            elif pii:
                reason = "pii"
            else:
                reason = "uncited"
        return {
            "id": str(uuid.uuid4()),
            "phase": interview["phase"],
            "question": question,
            "abstain": abstain,
            "abstainReason": reason,
            "reply": "" if abstain else reply,
            "evidenceIds": [] if abstain else evidence_ids,
            "retrievedIds": [item["id"] for item in evidence],
            "improved": False,
        }

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
