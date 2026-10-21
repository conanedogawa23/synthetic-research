import json
import os
import tempfile
import unittest
from pathlib import Path

from synthetic_research.chat import parse_model_json
from synthetic_research.corpus import campaign_sources, corpus_hash, validate_campaign
from synthetic_research.engine import ResearchEngine
from synthetic_research.gates import ResearchError, quote_is_verbatim
from synthetic_research.grounding import grounding_confidence, lock_citations
from synthetic_research.knowledge_mcp import graph_query, graph_upsert, vector_query, vector_upsert
from synthetic_research.mcp_client import ResearchMcp, first_public_https, public_https
from synthetic_research.prompts import RESPONDENT_SYSTEM
from synthetic_research.retrieve import build_index, retrieve
from synthetic_research.s3_log import _merge
from synthetic_research.store import SessionStore


PACK = {
    "campaignId": "CAM-1842",
    "grantId": "gold-1842",
    "kit": {"version": "v3", "locked": True, "brandAlias": "Aroha", "purpose": "RTD coffee"},
    "marketPack": {"market": "IN", "taboos": ["Do not mock filter coffee."]},
    "cohorts": [
        {
            "id": "office-sippers",
            "cardId": "R1",
            "label": "Office sipper",
            "rule": "Office sippers 25–34",
            "bounds": ["25–34", "not tea-only"],
            "traits": ["desk drinker"],
            "tension": "cafe taste / value price",
            "affinity": ["Bru"],
        }
    ],
    "guide": {
        "mains": [
            {"id": "m1", "phase": "rapport", "text": "Walk me through a weekday morning drink."},
            {"id": "m3", "phase": "unprompted", "text": "When you want cold coffee out of home, what comes to mind first?"},
        ],
        "probes": [{"id": "p1", "text": "What does that get you?", "when": "Any short answer"}],
    },
    "documents": [
        {
            "id": "T-14",
            "kind": "transcript",
            "cohortIds": ["office-sippers"],
            "title": "Bengaluru office",
            "text": "A can looks like I packed breakfast. I can drink it at the desk and nobody asks if I am on a diet.",
        },
        {
            "id": "R-882",
            "kind": "review",
            "cohortIds": ["office-sippers"],
            "title": "Medicinal can",
            "text": "Tastes like medicine. I wanted cold coffee and got a protein shake. Will not buy again.",
        },
        {
            "id": "S-01",
            "kind": "screener",
            "cohortIds": ["office-sippers"],
            "members": 12410,
            "title": "Office incidence",
            "text": "Member count in this gold slice is 12410. Tea-only respondents are a hard out.",
        },
    ],
}


class FakeLlm:
    def respondent(self, user: str) -> dict:
        payload = json.loads(user)
        evidence = payload["evidence"]
        snippet = evidence[0]["text"]
        return {"abstain": False, "reply": snippet, "evidenceIds": [evidence[0]["id"]]}

    def pack(self, user: str) -> dict:
        if not user.lstrip().startswith("{"):
            user = user[user.find("{") :]
        payload = json.loads(user)
        cohorts = []
        seen = set()
        for turn in payload["citedTurns"]:
            if turn["cohortId"] in seen:
                continue
            seen.add(turn["cohortId"])
            cohorts.append(
                {
                    "cohortId": turn["cohortId"],
                    "promise": "Cold coffee that looks like packed breakfast.",
                    "tension": "cafe taste / value price",
                    "themes": [
                        {
                            "theme": "Desk-safe",
                            "quote": turn["reply"][:48],
                            "evidenceIds": turn["evidenceIds"],
                        }
                    ],
                }
            )
        return {"cohorts": cohorts, "disclosureNote": "Exploratory. Not a population claim."}


class ResearchTests(unittest.TestCase):
    def test_retrieval_finds_the_medicinal_review(self):
        hits = retrieve(build_index(PACK["documents"]), "protein shake tastes like medicine", cohort_id="office-sippers")
        self.assertEqual(hits[0]["id"], "R-882")

    def test_bengaluru_filter_sales_are_an_additional_dataset(self):
        pack = json.loads((Path(__file__).resolve().parents[1] / "gold" / "cam-1842.json").read_text(encoding="utf-8"))
        validate_campaign(pack)
        self.assertEqual(pack["datasets"][0]["id"], "DS-BLR-FILTER")
        areas = {row["area"] for row in pack["datasets"][0]["rows"]}
        self.assertTrue(
            {"Whitefield", "Electronic City", "Koramangala", "Indiranagar", "MG Road", "Manyata Tech Park"} <= areas
        )
        sources = campaign_sources(pack)
        sales = [document for document in sources if document["kind"] == "sales"]
        self.assertEqual(len(sales), 10)
        self.assertEqual(len(sources), len(pack["documents"]) + len(sales))
        bare = {**pack, "datasets": []}
        self.assertNotEqual(corpus_hash(pack), corpus_hash(bare))
        hits = retrieve(
            build_index(sources),
            "filter coffee sales in Whitefield Electronic City office canteen",
            cohort_id="office-sippers",
        )
        self.assertIn(hits[0]["id"], {"SL-WF", "SL-EC"})
        self.assertEqual(hits[0]["kind"], "sales")
        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), FakeLlm())
            session = engine.open(pack, actor="Meera")
            granted = engine.grant(session["sessionId"], pack["grantId"], steward="Arjun")
            self.assertEqual(granted["grant"]["documentCount"], len(sources))
            cards = engine.compile_cards(session["sessionId"])
            office = next(card for card in cards if card["cohortId"] == "office-sippers")
            self.assertEqual(office["provenance"]["sales"], 10)

    def test_quote_must_be_verbatim(self):
        self.assertTrue(quote_is_verbatim("packed breakfast", ["A can looks like I packed breakfast."]))
        self.assertFalse(quote_is_verbatim("India loves cans", ["A can looks like I packed breakfast."]))

    def test_missing_evidence_answers_from_the_system_datasets(self):
        class RecordingLlm(FakeLlm):
            def respondent(self, user: str) -> dict:
                payload = json.loads(user)
                if payload.get("source") == "system":
                    self.datasets = payload["systemDatasets"]
                    return {
                        "abstain": True,
                        "reply": "I would still want the bitter cup at the office counter.",
                        "evidenceIds": ["SL-WF"],
                    }
                return super().respondent(user)

        pack = json.loads(json.dumps(PACK))
        pack["datasets"] = [
            {
                "id": "DS-BLR-FILTER",
                "label": "Filter coffee sales in Bengaluru office areas",
                "rows": [
                    {
                        "id": "SL-WF",
                        "cohortIds": ["office-sippers"],
                        "area": "Whitefield",
                        "cups": 18420,
                        "avgTicketInr": 30,
                        "text": "Whitefield office canteens sold filter coffee by the cup.",
                    }
                ],
            }
        ]
        llm = RecordingLlm()
        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), llm)
            session = engine.open(pack, actor="Meera")
            engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
            engine.compile_cards(session["sessionId"])
            engine.start_interview(session["sessionId"], "office-sippers")
            tea = engine.ask(session["sessionId"], "office-sippers", "What tea do you brew in the morning?")
            self.assertEqual(tea["abstainReason"], "out_of_screener_bounds")
            turn = engine.ask(session["sessionId"], "office-sippers", "Which railway timetable do you keep?")
        self.assertFalse(turn["abstain"])
        self.assertEqual(turn["source"], "system")
        self.assertEqual(turn["evidenceIds"], [])
        self.assertIn("No evidence was found", turn["reply"])
        self.assertIn("from the system", turn["reply"])
        self.assertEqual(llm.datasets[0]["datasetId"], "DS-BLR-FILTER")
        self.assertIn("Whitefield", llm.datasets[0]["text"])

    def test_uncited_indirect_reply_declines(self):
        class Abstain(FakeLlm):
            def respondent(self, user: str) -> dict:
                payload = json.loads(user)
                self.calls = getattr(self, "calls", 0) + 1
                self.source = payload.get("source")
                return {"abstain": True, "reply": "", "evidenceIds": []}

        pack = json.loads(json.dumps(PACK))
        pack["documents"] = [
            {
                "id": "R-882",
                "kind": "review",
                "cohortIds": ["office-sippers"],
                "title": "Medicinal can",
                "text": "The sweetness sits on top of a chalky note.",
            },
            pack["documents"][-1],
        ]
        llm = Abstain()
        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), llm)
            session = engine.open(pack, actor="Meera")
            engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
            engine.compile_cards(session["sessionId"])
            engine.start_interview(session["sessionId"], "office-sippers")
            turn = engine.ask(session["sessionId"], "office-sippers", "What did the sweetness get you?")
        self.assertEqual(llm.calls, 1)
        self.assertIsNone(llm.source)
        self.assertTrue(turn["abstain"])
        self.assertEqual(turn["abstainReason"], "uncited")
        self.assertNotIn("source", turn)
        self.assertEqual(turn["evidenceIds"], [])

    def test_web_mcp_tool_call_is_cited_when_the_campaign_has_no_span(self):
        class WebMcp:
            def gather(self, question: str, cohort_id: str, **_extra: object) -> list[dict]:
                del question, cohort_id, _extra
                return [
                    {
                        "database": "web",
                        "server": "fetch",
                        "tool": "fetch",
                        "ok": True,
                        "text": "The office filter counter is the weekday cup. A cafe cup costs more.",
                        "url": "https://example.com/filter-coffee",
                        "error": "",
                    },
                    {
                        "database": "vector",
                        "server": "qdrant",
                        "tool": "qdrant-find",
                        "ok": False,
                        "text": "",
                        "url": "",
                        "error": "MCP_VECTOR is off.",
                    },
                ]

        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), FakeLlm(), mcp=WebMcp())
            session = engine.open(PACK, actor="Meera")
            engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
            engine.compile_cards(session["sessionId"])
            engine.start_interview(session["sessionId"], "office-sippers")
            turn = engine.ask(session["sessionId"], "office-sippers", "Which railway timetable do you keep?")
            chip = engine.open_evidence(session["sessionId"], "WEB-1")
        self.assertFalse(turn["abstain"])
        self.assertEqual(turn["evidenceIds"], ["WEB-1"])
        self.assertEqual(turn["confidence"]["basis"], "cited score from web")
        self.assertEqual(turn["mcp"][0]["tool"], "fetch")
        self.assertFalse(turn["mcp"][1]["ok"])
        self.assertIn("cafe cup costs more", chip["text"])

    def test_mcp_catalog_stays_quiet_until_a_database_is_configured(self):
        self.assertFalse(public_https("http://example.com/page"))
        self.assertFalse(public_https("https://127.0.0.1/secret"))
        self.assertEqual(first_public_https("see https://example.com/filter."), "https://example.com/filter")
        keys = (
            "MCP_WEB",
            "MCP_VECTOR",
            "MCP_GRAPH",
            "BRAVE_API_KEY",
            "MCP_FETCH_URL",
            "QDRANT_URL",
            "QDRANT_LOCAL_PATH",
            "NEO4J_MCP_PASSWORD",
            "KNOWLEDGE_MCP_DIR",
        )
        saved = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            results = ResearchMcp().gather("filter coffee in Bengaluru", "office-sippers")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual([item["database"] for item in results], ["web", "vector", "graph"])
        self.assertTrue(all(not item["ok"] for item in results))
        self.assertEqual(results[0]["tool"], "search")

    def test_web_search_uses_duckduckgo_without_an_api_key(self):
        keys = (
            "MCP_WEB",
            "BRAVE_API_KEY",
            "MCP_FETCH_URL",
            "DDG_REGION",
            "MCP_VECTOR",
            "MCP_GRAPH",
            "KNOWLEDGE_MCP_DIR",
            "QDRANT_URL",
            "QDRANT_LOCAL_PATH",
            "NEO4J_MCP_PASSWORD",
        )
        saved = {key: os.environ.get(key) for key in keys}
        os.environ["MCP_WEB"] = "true"
        os.environ["DDG_REGION"] = "in-en"
        for key in keys:
            if key not in {"MCP_WEB", "DDG_REGION"}:
                os.environ.pop(key, None)
        calls = []

        def fake_call(server, tool, arguments):
            calls.append((server.name, server.args, tool, arguments, server.env))
            if tool == "search":
                return "Filter coffee\nhttps://example.com/filter-coffee\nA cafe cup costs more."
            return "The office filter counter is the weekday cup."

        try:
            client = ResearchMcp()
            client.call_tool = fake_call
            results = client.gather("filter coffee in Bengaluru", "office-sippers")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(calls[0][0], "duckduckgo")
        self.assertIn("duckduckgo-mcp-server[browser]", calls[0][1])
        self.assertEqual(calls[0][2], "search")
        self.assertNotIn("BRAVE_API_KEY", calls[0][4])
        self.assertEqual(calls[1][2], "fetch_content")
        self.assertEqual(calls[1][3]["url"], "https://example.com/filter-coffee")
        self.assertTrue(results[0]["ok"])
        self.assertEqual(results[0]["tool"], "search")
        self.assertTrue(results[1]["ok"])

    def test_session_knowledge_mcp_stores_and_answers(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp is installed in .venv")

        keys = (
            "KNOWLEDGE_MCP_DIR",
            "MCP_VECTOR",
            "MCP_GRAPH",
            "QDRANT_URL",
            "QDRANT_LOCAL_PATH",
            "NEO4J_MCP_PASSWORD",
        )
        saved = {key: os.environ.get(key) for key in keys}
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["KNOWLEDGE_MCP_DIR"] = tmp
            try:
                session_id = "a9cbd64f-f3d8-45c9-8141-c8fde0c84e4d"
                vector_upsert(
                    session_id,
                    json.dumps(
                        [
                            {
                                "id": "UP-001-c1",
                                "text": "Profitable growth protects margins and sales productivity.",
                                "vector": [1.0, 0.0],
                            }
                        ]
                    ),
                )
                found = vector_query(session_id, "What protects margins?", json.dumps([1.0, 0.0]), 2)
                self.assertIn("UP-001-c1", found)
                graph_upsert(
                    session_id,
                    json.dumps(
                        {
                            "nodes": [],
                            "edges": [{"from": "Frontline", "relation": "supports", "to": "field teams"}],
                        }
                    ),
                )
                linked = graph_query(session_id, "Who supports the field teams?")
                self.assertIn("Frontline supports field teams", linked)
                client = ResearchMcp()
                os.environ["MCP_VECTOR"] = "true"
                os.environ["MCP_GRAPH"] = "true"
                os.environ.pop("QDRANT_URL", None)
                os.environ.pop("QDRANT_LOCAL_PATH", None)
                os.environ.pop("NEO4J_MCP_PASSWORD", None)
                stored = client.index_knowledge(
                    session_id,
                    [{"id": "UP-001-c2", "text": "Merchandising productivity is the weekly check.", "vector": [0.0, 1.0]}],
                    {"nodes": [], "edges": [{"from": "Merchandising", "relation": "is", "to": "weekly"}]},
                )
                self.assertTrue(all(row["ok"] for row in stored), stored)
                asked = client.gather("What is the weekly check?", "respondents", session_id=session_id, query_vector=[0.0, 1.0])
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        by_database = {row["database"]: row for row in asked}
        self.assertTrue(by_database["vector"]["ok"], by_database["vector"])
        self.assertIn("UP-001-c2", by_database["vector"]["text"])
        self.assertEqual(by_database["vector"]["tool"], "vector_query")
        self.assertTrue(by_database["graph"]["ok"], by_database["graph"])
        self.assertEqual(by_database["graph"]["tool"], "graph_query")
        self.assertFalse(by_database["web"]["ok"])

    def test_one_session_log_collects_each_question(self):
        first = _merge({}, {"sessionId": "s1", "campaignId": "c1", "turnId": "t1", "question": "First"})
        second = _merge(first, {"sessionId": "s1", "campaignId": "c1", "turnId": "t2", "question": "Second"})
        replaced = _merge(second, {"sessionId": "s1", "campaignId": "c1", "turnId": "t1", "question": "First, improved"})
        self.assertEqual([item["question"] for item in replaced["questions"]], ["First, improved", "Second"])
        self.assertEqual(replaced["sessionId"], "s1")

    def test_a_reply_cannot_keep_a_document_it_was_not_given(self):
        reply, kept, removed = lock_citations(
            "winning where we have the right to win [UP-009-c1, UP-003-c2].",
            ["UP-003-c2", "WEB-4", "GRAPH-7"],
            {"UP-003-c2", "WEB-4", "GRAPH-7"},
        )
        self.assertEqual(removed, ["UP-009-c1"])
        self.assertEqual(reply, "winning where we have the right to win.")
        self.assertNotIn("UP-003-c2", reply)
        self.assertIn("UP-003-c2", kept)
        midsentence, midsentence_kept, _ = lock_citations(
            "The approach breaks down [UP-003-c2] because the data is split.",
            ["UP-003-c2"],
            {"UP-003-c2"},
        )
        self.assertEqual(midsentence, "The approach breaks down because the data is split.")
        self.assertEqual(midsentence_kept, ["UP-003-c2"])
        self.assertNotIn("[,", reply)
        self.assertNotIn("GRAPH-7", kept)
        self.assertNotIn("WEB-4", kept)
        self.assertEqual(grounding_confidence(supported=False, removed=removed, used_web=False)["label"], "medium")
        self.assertEqual(grounding_confidence(supported=True, removed=[], used_web=False)["score"], 1.0)

    def test_respondent_treats_a_cafe_contrast_as_an_answer(self):
        self.assertIn("contrast inside a span is an answer", RESPONDENT_SYSTEM)
        self.assertIn("contrast inside a span is an answer", RESPONDENT_SYSTEM)

    def test_json_fence_parses(self):
        parsed = parse_model_json('```json\n{"ok": true}\n```')
        self.assertTrue(parsed["ok"])

    def test_session_rules_and_vault(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), FakeLlm())
            session = engine.open(PACK, actor="Meera")
            self.assertTrue(session["insightLocked"])
            with self.assertRaises(ResearchError) as locked:
                engine.list_cohorts(session["sessionId"])
            self.assertEqual(locked.exception.code, "INSIGHT_LOCKED")

            engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
            cohorts = engine.list_cohorts(session["sessionId"])
            self.assertEqual(cohorts[0]["members"], 12410)
            engine.compile_cards(session["sessionId"])
            with self.assertRaises(ResearchError) as main:
                engine.edit_main()
            self.assertEqual(main.exception.code, "MAIN_LOCKED")
            with self.assertRaises(ResearchError) as cohort:
                engine.add_cohort()
            self.assertEqual(cohort.exception.code, "COHORT_LOCKED")

            engine.start_interview(session["sessionId"], "office-sippers")
            with self.assertRaises(ResearchError) as brand:
                engine.ask(session["sessionId"], "office-sippers", "What do you think of Aroha?")
            self.assertEqual(brand.exception.code, "BRAND_LOCKED")
            tea = engine.ask(session["sessionId"], "office-sippers", "What tea do you brew in the morning?")
            self.assertTrue(tea["abstain"])
            self.assertEqual(tea["abstainReason"], "out_of_screener_bounds")
            cited = engine.ask(session["sessionId"], "office-sippers", "Walk me through a weekday morning drink.")
            self.assertFalse(cited["abstain"])
            self.assertIn(cited["evidenceIds"][0], {"T-14", "R-882", "S-01"})
            chip = engine.open_evidence(session["sessionId"], cited["evidenceIds"][0])
            self.assertNotIn("email", chip)
            engine.advance_phase(session["sessionId"], "office-sippers")
            engine.ask(session["sessionId"], "office-sippers", "When you want cold coffee out of home, what comes to mind first?")
            engine.end_interview(session["sessionId"], "office-sippers")
            with self.assertRaises(ResearchError) as missing_skip:
                engine.approve(session["sessionId"], "Meera Iyer")
            self.assertIn(missing_skip.exception.code, {"PACK_MISSING", "SKIP_MISSING"})
            engine.draft_pack(session["sessionId"])
            engine.approve(session["sessionId"], "Meera Iyer")
            vault = engine.store.read_vault(session["sessionId"])
            self.assertEqual(vault["badge"], "exploratory")
            self.assertNotIn("documents", vault)
            self.assertEqual(vault["oversightName"], "Meera Iyer")

            engine.refresh(session["sessionId"], steward="Arjun")
            with self.assertRaises(ResearchError) as stale:
                engine.draft_pack(session["sessionId"])
            self.assertEqual(stale.exception.code, "PACK_BLOCKED")
            self.assertIsNotNone(engine.store.read_vault(session["sessionId"]))

            engine.close(session["sessionId"])
            self.assertFalse(engine.store.index_exists(session["sessionId"]))
            other = engine.open(PACK, actor="Meera")
            with self.assertRaises(ResearchError) as isolated:
                engine.list_cohorts(other["sessionId"])
            self.assertEqual(isolated.exception.code, "INSIGHT_LOCKED")
            self.assertFalse(engine.store.index_exists(other["sessionId"]))
            actions = [event["action"] for event in engine.store.read_audit(session["sessionId"])]
            self.assertIn("SESSION_OPENED", actions)
            self.assertIn("GRANT_ACTIVATED", actions)
            self.assertIn("PACK_APPROVED", actions)
            self.assertIn("SESSION_CLOSED", actions)
            self.assertTrue(engine.store.read(session["sessionId"])["corpusHash"])

    def test_pack_rejects_contact_data_and_a_broken_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), FakeLlm())
            broken = {**PACK, "campaignId": ""}
            with self.assertRaises(ResearchError) as invalid:
                engine.open(broken, actor="Meera")
            self.assertEqual(invalid.exception.code, "PACK_INVALID")
            session = engine.open(PACK, actor="Meera")
            tainted = json.loads(json.dumps(PACK))
            tainted["documents"][0]["text"] += " Contact maker@example.com for the recording."
            engine.store.read(session["sessionId"])
            stored = engine.store.read(session["sessionId"])
            stored["documents"] = tainted["documents"]
            engine.store.write(stored)
            with self.assertRaises(ResearchError) as pii:
                engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
            self.assertEqual(pii.exception.code, "PII_REJECTED")

    def test_hybrid_search_ranks_meaning_when_the_words_do_not_overlap(self):
        from synthetic_research.search import retrieve_evidence

        documents = [
            {
                "id": "T-06",
                "kind": "transcript",
                "cohortIds": ["office-sippers"],
                "title": "Delhi",
                "text": "My flatmate is asleep and the morning drink is mine.",
                "vector": [1.0, 0.0],
            },
            {
                "id": "R-882",
                "kind": "review",
                "cohortIds": ["office-sippers"],
                "title": "Medicinal",
                "text": "Tastes like medicine and a protein shake.",
                "vector": [0.0, 1.0],
            },
        ]
        hits = retrieve_evidence(
            documents,
            "who else is in that moment with you",
            cohort_id="office-sippers",
            query_vector=[0.99, 0.01],
        )
        self.assertEqual(hits[0]["id"], "T-06")

    def test_follow_up_runs_once_after_a_short_cited_reply(self):
        from synthetic_research.engine import should_follow_up

        cited = {"abstain": False, "reply": "I want it bitter enough to count as coffee."}
        self.assertTrue(should_follow_up(cited, 8))
        self.assertFalse(should_follow_up({"abstain": True, "reply": ""}, 8))
        self.assertFalse(should_follow_up(cited, 0))


if __name__ == "__main__":
    unittest.main()
