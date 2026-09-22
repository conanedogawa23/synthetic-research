import json
import tempfile
import unittest
from pathlib import Path

from synthetic_research.deepinfra import parse_model_json
from synthetic_research.engine import ResearchEngine
from synthetic_research.gates import ResearchError, quote_is_verbatim
from synthetic_research.retrieve import build_index, retrieve
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

    def test_quote_must_be_verbatim(self):
        self.assertTrue(quote_is_verbatim("packed breakfast", ["A can looks like I packed breakfast."]))
        self.assertFalse(quote_is_verbatim("India loves cans", ["A can looks like I packed breakfast."]))

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


if __name__ == "__main__":
    unittest.main()
