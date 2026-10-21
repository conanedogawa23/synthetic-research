import json
import tempfile
import unittest
from pathlib import Path

from synthetic_research.corpus import campaign_sources
from synthetic_research.prompts import PACK_SYSTEM, RESPONDENT_SYSTEM
from synthetic_research.engine import ResearchEngine
from synthetic_research.store import SessionStore
from synthetic_research.synthesis import format_synthesis, plan_synthesis


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
        "mains": [{"id": "m1", "phase": "rapport", "text": "Walk me through a weekday morning drink."}],
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
        return {"abstain": False, "reply": evidence[0]["text"], "evidenceIds": [evidence[0]["id"]]}

    def pack(self, user: str) -> dict:
        del user
        return {"cohorts": [], "disclosureNote": "Exploratory. Not a population claim."}


def _doc(document_id: str, text: str, kind: str = "transcript") -> dict:
    return {
        "id": document_id,
        "kind": kind,
        "cohortIds": ["office-sippers"],
        "title": document_id,
        "text": text,
    }


class SynthesisPlanTests(unittest.TestCase):
    def test_the_more_direct_document_is_chosen(self):
        loose = _doc(
            "R-loose",
            "Ninety rupees for a can and one hundred forty for the cafe downstairs.",
            kind="review",
        )
        direct = _doc(
            "R-210",
            "I will pay cafe money only when the can tastes like coffee rather than flavoured milk.",
            kind="review",
        )
        plan = plan_synthesis(
            "Would you pay cafe money for a can?",
            [loose, direct],
            retrieved_ids={"R-loose", "R-210"},
        )
        self.assertEqual(plan["mode"], "direct")
        self.assertEqual(plan["directDocumentId"], "R-210")
        self.assertEqual(plan["evidenceIds"][0], "R-210")
        self.assertIn("pay cafe money", plan["reply"])
        relations = {row["id"]: row["relation"] for row in plan["ranked"]}
        self.assertEqual(relations["R-210"], "direct")
        self.assertEqual(relations["R-loose"], "indirect")

    def test_a_reference_without_the_answer_stays_indirect(self):
        plan = plan_synthesis(
            "What did the sweetness get you?",
            [_doc("R-882", "The sweetness sits on top of a chalky note.")],
            retrieved_ids={"R-882"},
        )
        self.assertEqual(plan["mode"], "indirect")
        self.assertIsNone(plan["directDocumentId"])
        self.assertEqual(plan["reply"], "")
        self.assertEqual(plan["ranked"][0]["relation"], "indirect")

    def test_one_transcript_holding_both_spend_stances_is_not_a_group_conflict(self):
        plan = plan_synthesis(
            "Would you pay cafe money for a can?",
            [
                _doc(
                    "R-210",
                    "The can loses. I will pay cafe money only when it tastes like coffee. Until then I buy the offer price or I skip.",
                )
            ],
        )
        self.assertEqual(plan["mode"], "direct")
        self.assertEqual(plan["conflicts"], [])
        self.assertEqual(plan["directDocumentId"], "R-210")

    def test_negated_pay_is_refusal_not_willingness(self):
        plan = plan_synthesis(
            "Would you pay cafe money for a can?",
            [
                _doc("T-14", "I will not pay cafe money for a can."),
                _doc("T-31", "I would pay cafe money for a can that tastes like filter coffee."),
            ],
        )
        self.assertEqual(plan["mode"], "conflict")
        spend = plan["conflicts"][0]
        self.assertEqual(spend["point"], "spend")
        sides = {side["stance"]: side for side in spend["sides"]}
        self.assertEqual(sides["refuse"]["documentIds"], ["T-14"])
        self.assertEqual(sides["willing"]["documentIds"], ["T-31"])
        self.assertIn("will not pay", plan["reply"])
        self.assertIn("would pay", plan["reply"])
        self.assertIn("T-14", plan["evidenceIds"])
        self.assertIn("T-31", plan["evidenceIds"])
        self.assertGreater(len(plan["evidenceIds"]), 1)

    def test_an_implied_brewer_conflict_stays_with_the_model(self):
        plan = plan_synthesis(
            "Who brews the coffee?",
            [
                _doc("T-06", "The morning drink is mine."),
                _doc("T-44", "My mother makes coffee before I leave."),
                _doc("T-31", "At the office canteen I still look for filter."),
            ],
        )
        self.assertEqual(plan["mode"], "indirect")
        self.assertEqual(plan["reply"], "")
        self.assertEqual(plan["conflicts"][0]["point"], "brewer")
        self.assertIn("morning drink is mine", plan["fallbackReply"])
        self.assertIn("My mother", plan["fallbackReply"])
        self.assertIn("canteen", plan["fallbackReply"])

    def test_brewer_conflict_keeps_each_person(self):
        plan = plan_synthesis(
            "Who brews the coffee?",
            [
                _doc("T-06", "I brew the morning coffee. The morning drink is mine."),
                _doc("T-44", "My mother makes coffee before I leave."),
                _doc("T-31", "At the office canteen they brew filter."),
            ],
        )
        self.assertEqual(plan["mode"], "conflict")
        self.assertEqual(plan["conflicts"][0]["point"], "brewer")
        stances = {side["stance"] for side in plan["conflicts"][0]["sides"]}
        self.assertEqual(stances, {"self", "household", "counter"})
        self.assertIn("I brew the morning coffee", plan["reply"])
        self.assertIn("My mother", plan["reply"])
        self.assertIn("canteen", plan["reply"])

    def test_routine_conflict_keeps_the_refusal(self):
        plan = plan_synthesis(
            "Is a weekday routine allowed?",
            [
                _doc("T-14", "Weekday morning is Bru at the flat, then a cab."),
                _doc("T-22", "A treat is allowed on Sunday. A daily routine is not allowed."),
            ],
        )
        self.assertEqual(plan["mode"], "conflict")
        self.assertEqual(plan["conflicts"][0]["label"], "routine not allowed")
        self.assertIn("Weekday morning", plan["reply"])
        self.assertIn("not allowed", plan["reply"])

    def test_format_names_the_direct_document_and_the_sides(self):
        plan = plan_synthesis(
            "Would you pay cafe money for a can?",
            [
                _doc("T-14", "I will not pay cafe money for a can."),
                _doc("T-31", "I would pay cafe money for a can that tastes like filter coffee."),
            ],
        )
        lines = format_synthesis(
            {
                "mode": plan["mode"],
                "directDocumentId": plan["directDocumentId"],
                "conflicts": plan["conflicts"],
            }
        )
        self.assertTrue(any(plan["directDocumentId"] in line for line in lines))
        self.assertTrue(any("do not want to spend" in line and "T-14" in line for line in lines))
        self.assertTrue(any("would spend" in line and "T-31" in line for line in lines))


class SynthesisEngineTests(unittest.TestCase):
    def _engine(self, tmp: str, llm, pack: dict | None = None):
        engine = ResearchEngine(SessionStore(Path(tmp)), llm)
        session = engine.open(pack or PACK, actor="Meera")
        engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
        engine.compile_cards(session["sessionId"])
        engine.start_interview(session["sessionId"], "office-sippers")
        return engine, session["sessionId"]

    def test_direct_answer_is_one_persona_call(self):
        class Spoke(FakeLlm):
            def respondent(self, user: str) -> dict:
                self.calls = getattr(self, "calls", 0) + 1
                payload = json.loads(user)
                self.required = payload.get("requiredStatements")
                return {
                    "abstain": False,
                    "reply": "I start with whatever is already on the desk.",
                    "evidenceIds": [],
                }

        pack = json.loads(json.dumps(PACK))
        pack["documents"] = [
            _doc("T-14", "Weekday morning is Bru at the flat, then a cab."),
            pack["documents"][-1],
        ]
        llm = Spoke()
        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, llm, pack)
            turn = engine.ask(session_id, "office-sippers", "What is the weekday morning drink?")
        self.assertEqual(llm.calls, 1)
        self.assertTrue(llm.required)
        self.assertFalse(turn["abstain"])
        self.assertEqual(turn["synthesis"]["mode"], "direct")
        self.assertEqual(turn["synthesis"]["directDocumentId"], "T-14")
        self.assertEqual(turn["evidenceIds"], ["T-14"])
        self.assertIn("already on the desk", turn["reply"])
        self.assertIn("Weekday morning", turn["reply"])
        self.assertIn("direct answer in T-14", turn["confidence"]["basis"])

    def test_implied_conflict_uses_the_model_when_every_side_is_cited(self):
        class Citing(FakeLlm):
            def respondent(self, user: str) -> dict:
                payload = json.loads(user)
                ids = [
                    side["excerptDocumentId"]
                    for conflict in payload["synthesis"]["conflicts"]
                    for side in conflict["sides"]
                ]
                return {"abstain": False, "reply": "It depends on who brews it.", "evidenceIds": ids}

        class OneSide(FakeLlm):
            def respondent(self, user: str) -> dict:
                payload = json.loads(user)
                return {
                    "abstain": False,
                    "reply": "The morning drink is mine.",
                    "evidenceIds": [payload["evidence"][0]["id"]],
                }

        pack = json.loads(json.dumps(PACK))
        pack["documents"] = [
            _doc("T-06", "The morning drink is mine."),
            _doc("T-44", "My mother makes coffee before I leave."),
            _doc("T-31", "At the office canteen I still look for filter."),
            pack["documents"][-1],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, Citing(), pack)
            kept = engine.ask(session_id, "office-sippers", "Who brews the coffee?")
        self.assertEqual(kept["synthesis"]["mode"], "indirect")
        self.assertIn("It depends on who brews it.", kept["reply"])
        self.assertIn("The morning drink is mine.", kept["reply"])
        self.assertIn("My mother makes coffee before I leave.", kept["reply"])
        self.assertIn("At the office canteen I still look for filter.", kept["reply"])
        self.assertEqual(kept["synthesis"]["conflicts"][0]["point"], "brewer")
        self.assertTrue({"T-06", "T-44", "T-31"} <= set(kept["evidenceIds"]))

        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, OneSide(), pack)
            restored = engine.ask(session_id, "office-sippers", "Who brews the coffee?")
        self.assertEqual(restored["synthesis"]["mode"], "indirect")
        self.assertEqual(restored["reply"], "The morning drink is mine.")
        self.assertIn("T-06", restored["evidenceIds"])

    def test_the_next_question_receives_the_previous_reply(self):
        class Recording(FakeLlm):
            def respondent(self, user: str) -> dict:
                payload = json.loads(user)
                self.prior = payload.get("priorTurns") or []
                return super().respondent(user)

        pack = json.loads(json.dumps(PACK))
        pack["documents"] = [
            _doc("T-14", "Weekday morning is Bru at the flat, then a cab."),
            pack["documents"][-1],
        ]
        llm = Recording()
        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, llm, pack)
            engine.ask(session_id, "office-sippers", "What is the weekday morning drink?")
            engine.ask(session_id, "office-sippers", "What does the cab change?")
        self.assertEqual(llm.prior[-1]["question"], "What is the weekday morning drink?")
        self.assertIn("Weekday morning", llm.prior[-1]["reply"])

    def test_indirect_reference_is_inferred_by_the_model(self):
        class Recording(FakeLlm):
            def respondent(self, user: str) -> dict:
                payload = json.loads(user)
                self.task = payload.get("task")
                self.mode = payload.get("synthesis", {}).get("mode")
                return {"abstain": False, "reply": "The chalky note is why I dropped it.", "evidenceIds": ["R-882"]}

        pack = json.loads(json.dumps(PACK))
        pack["documents"] = [
            _doc("R-882", "The sweetness sits on top of a chalky note.", kind="review"),
            pack["documents"][-1],
        ]
        llm = Recording()
        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, llm, pack)
            turn = engine.ask(session_id, "office-sippers", "What did the sweetness get you?")
        self.assertEqual(llm.task, "indirect")
        self.assertEqual(llm.mode, "indirect")
        self.assertEqual(turn["synthesis"]["mode"], "indirect")
        self.assertIsNone(turn["synthesis"]["directDocumentId"])
        self.assertEqual(turn["evidenceIds"], ["R-882"])

    def test_conflict_is_quoted_and_a_smoothing_rewrite_is_rejected(self):
        class Smoothing(FakeLlm):
            calls = 0

            def respondent(self, user: str) -> dict:
                self.calls += 1
                payload = json.loads(user)
                return {
                    "abstain": False,
                    "reply": "I would pay cafe money and I am fine with the price.",
                    "evidenceIds": [payload["evidence"][0]["id"]],
                }

        pack = json.loads(json.dumps(PACK))
        pack["documents"] = [
            _doc("T-14", "I will not pay cafe money for a can."),
            _doc("T-31", "I would pay cafe money for a can that tastes like filter coffee."),
            pack["documents"][-1],
        ]
        llm = Smoothing()
        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, llm, pack)
            turn = engine.ask(session_id, "office-sippers", "Would you pay cafe money for a can?")
            self.assertEqual(llm.calls, 1)
            improved = engine.improve(session_id, "office-sippers", "Make this one preference.")
        self.assertEqual(turn["synthesis"]["mode"], "conflict")
        self.assertEqual(turn["synthesis"]["conflicts"][0]["point"], "spend")
        self.assertIn("I would pay cafe money and I am fine with the price.", turn["reply"])
        self.assertIn("will not pay", turn["reply"])
        self.assertIn("would pay", turn["reply"])
        self.assertEqual(improved["reply"], turn["reply"])
        self.assertFalse(improved["improved"])
        self.assertEqual(llm.calls, 2)
        self.assertEqual(turn["confidence"]["basis"], "direct answers in conflict")

    def test_gold_spend_question_names_the_direct_document_and_both_sides(self):
        pack = json.loads((Path(__file__).resolve().parents[1] / "gold" / "cam-1842.json").read_text(encoding="utf-8"))
        group = [
            document
            for document in campaign_sources(pack)
            if "office-sippers" in (document.get("cohortIds") or [])
        ]
        plan = plan_synthesis("What would make you pay cafe money for a can?", group)
        self.assertEqual(plan["mode"], "conflict")
        spend = next(conflict for conflict in plan["conflicts"] if conflict["point"] == "spend")
        sides = {side["stance"] for side in spend["sides"]}
        self.assertIn("refuse", sides)
        self.assertIn("willing", sides)
        winner = next(document for document in group if document["id"] == plan["directDocumentId"])
        self.assertIn(plan["reply"].split(". ")[0], winner["text"])

        class Spoke(FakeLlm):
            def respondent(self, user: str) -> dict:
                self.calls = getattr(self, "calls", 0) + 1
                return {
                    "abstain": False,
                    "reply": "The price is the whole argument for me.",
                    "evidenceIds": [],
                }

        llm = Spoke()
        with tempfile.TemporaryDirectory() as tmp:
            engine, session_id = self._engine(tmp, llm, pack)
            turn = engine.ask(session_id, "office-sippers", "What would make you pay cafe money for a can?")
        self.assertEqual(llm.calls, 1)
        self.assertEqual(turn["synthesis"]["directDocumentId"], plan["directDocumentId"])
        self.assertEqual(turn["evidenceIds"][0], plan["directDocumentId"])
        self.assertIn("The price is the whole argument for me.", turn["reply"])
        self.assertFalse(turn["abstain"])

    def test_a_web_page_is_not_pasted_when_it_only_matches_the_wording(self):
        plan = plan_synthesis(
            "what are your biggest challenges in delivering your priorities",
            [
                {
                    "id": "WEB-10",
                    "kind": "web",
                    "text": (
                        "Skip to main content. Sign in to view more content. "
                        "What are the most common challenges in managing priorities?"
                    ),
                },
                {
                    "id": "UP-008-c1",
                    "kind": "transcript",
                    "text": (
                        "Country sales teams translate regional and global priorities into local plans "
                        "and are responsible for delivering sales targets."
                    ),
                },
            ],
            retrieved_ids={"WEB-10", "UP-008-c1"},
        )
        self.assertNotEqual(plan["directDocumentId"], "WEB-10")
        self.assertNotIn("Skip to main content", plan.get("reply") or "")
        self.assertEqual(plan["mode"], "indirect")

    def test_prompt_keeps_indirect_inference_and_conflict_sides_for_the_model(self):
        self.assertIn("When task is indirect", RESPONDENT_SYSTEM)
        self.assertIn("When requiredStatements is not empty", RESPONDENT_SYSTEM)
        self.assertIn("priorCommitments", RESPONDENT_SYSTEM)
        self.assertIn("Do not merge a recorded conflict into one preference", RESPONDENT_SYSTEM)
        self.assertIn("If a cited turn lists conflicts", PACK_SYSTEM)


if __name__ == "__main__":
    unittest.main()
