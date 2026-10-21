import json
import tempfile
import unittest
from pathlib import Path

from synthetic_research.api import ResearchApi, load_packs
from synthetic_research.engine import ResearchEngine
from synthetic_research.store import SessionStore


class FakeLlm:
    def respondent(self, user: str) -> dict:
        payload = json.loads(user)
        if payload.get("source") == "system":
            return {"abstain": False, "reply": "I keep a can for the cab.", "evidenceIds": []}
        evidence = payload["evidence"]
        return {"abstain": False, "reply": evidence[0]["text"][:160], "evidenceIds": [evidence[0]["id"]]}

    def follow_up(self, user: str) -> dict:
        del user
        return {"question": "What does that get you?"}

    def pack(self, user: str) -> dict:
        payload = json.loads(user[user.find("{") :])
        turn = payload["citedTurns"][0]
        return {
            "disclosureNote": "Exploratory. Not a population claim.",
            "cohorts": [
                {
                    "cohortId": turn["cohortId"],
                    "promise": "Cold coffee that looks like packed breakfast.",
                    "tension": "cafe taste / value price",
                    "themes": [
                        {
                            "theme": "Desk-safe",
                            "quote": turn["reply"][:48] if len(turn["reply"]) >= 12 else turn["reply"],
                            "evidenceIds": turn["evidenceIds"],
                        }
                    ],
                }
            ],
        }


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class ResearchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        gold = Path(__file__).resolve().parents[1] / "gold"
        self.api = ResearchApi(
            ResearchEngine(SessionStore(Path(self.tmp.name)), FakeLlm()),
            load_packs(gold),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_session_can_be_granted_and_asked_from_the_http_contract(self) -> None:
        campaigns = self.api.handle("GET", "/campaigns", None, "Meera")
        self.assertTrue(campaigns["ok"])
        self.assertIn("CAM-1842", [campaign["id"] for campaign in campaigns["data"]])
        created_campaign = self.api.handle(
            "POST",
            "/campaigns",
            {"name": "Office coffee trial", "market": "India", "category": "Ready-to-drink coffee"},
            "Meera",
        )
        self.assertTrue(created_campaign["ok"], created_campaign)
        self.assertEqual(created_campaign["data"]["name"], "Office coffee trial")
        opened = self.api.handle(
            "POST",
            f"/campaigns/{created_campaign['data']['id']}/sessions",
            {},
            "Meera",
        )
        self.assertTrue(opened["ok"], opened)
        self.assertEqual(opened["data"]["campaignId"], created_campaign["data"]["id"])

        created = self.api.handle("POST", "/campaigns/CAM-1842/sessions", {}, "Meera")
        self.assertTrue(created["ok"])
        session_id = created["data"]["id"]
        self.assertEqual(created["data"]["stage"], "intake")
        self.assertFalse(created["data"]["evidenceReady"])

        missing = self.api.handle("POST", f"/sessions/{session_id}/grant", {}, "Meera")
        self.assertEqual(missing["code"], "STEWARD_MISSING")

        granted = self.api.handle("POST", f"/sessions/{session_id}/grant", {"steward": "Arjun"}, "Meera")
        self.assertTrue(granted["ok"])
        self.assertEqual(granted["data"]["grantId"], "gold-1842")
        self.assertTrue(granted["data"]["evidenceReady"])

        cards = self.api.handle("GET", f"/sessions/{session_id}/respondents", None, "Meera")
        self.assertTrue(cards["ok"])
        office = next(card for card in cards["data"] if card["cohortId"] == "office-sippers")

        interview = self.api.handle(
            "POST",
            f"/sessions/{session_id}/interviews",
            {"cardId": office["id"]},
            "Meera",
        )
        self.assertEqual(interview["data"]["phase"], "rapport")
        interview_id = interview["data"]["id"]

        blocked = self.api.handle(
            "POST",
            f"/sessions/{session_id}/interviews/{interview_id}/questions",
            {"question": "What do you think of Aroha?"},
            "Meera",
        )
        self.assertEqual(blocked["code"], "BRAND_LOCKED")

        tea = self.api.handle(
            "POST",
            f"/sessions/{session_id}/interviews/{interview_id}/questions",
            {"question": "What tea do you brew in the morning?"},
            "Meera",
        )
        self.assertEqual(tea["data"]["reply"]["kind"], "abstain")

        answer = self.api.handle(
            "POST",
            f"/sessions/{session_id}/interviews/{interview_id}/questions",
            {"question": "When you want cold coffee out of home, what comes to mind first?"},
            "Meera",
        )
        self.assertTrue(answer["ok"], answer)
        self.assertIn(answer["data"]["reply"]["kind"], {"cited", "system"})
        if answer["data"]["reply"]["kind"] == "cited":
            span_id = answer["data"]["reply"]["chips"][0]["spanId"]
            chip = self.api.handle("GET", f"/sessions/{session_id}/evidence/{span_id}", None, "Meera")
            self.assertTrue(chip["ok"])
            self.assertTrue(chip["data"]["text"])
            self.assertNotIn("email", chip["data"])

    def test_uploaded_knowledge_builds_vectors_and_a_graph(self) -> None:
        import base64

        from synthetic_research.api import ResearchApi, load_packs
        from synthetic_research.engine import ResearchEngine
        from synthetic_research.store import SessionStore

        api = ResearchApi(
            ResearchEngine(SessionStore(Path(self.tmp.name) / "knowledge"), FakeLlm(), embedder=FakeEmbedder()),
            load_packs(Path(__file__).resolve().parents[1] / "gold"),
        )
        created = api.handle("POST", "/campaigns/CAM-1842/sessions", {}, "Meera")
        session_id = created["data"]["id"]
        self.assertFalse(created["data"]["knowledge"]["ready"])
        uploaded = api.handle(
            "POST",
            f"/sessions/{session_id}/knowledge",
            {
                "filename": "evergreen.txt",
                "contentBase64": base64.b64encode(
                    b"Evergreen uses Frontline for store execution. Frontline supports the field teams."
                ).decode("ascii"),
            },
            "Meera",
        )
        self.assertTrue(uploaded["ok"], uploaded)
        self.assertEqual(uploaded["data"]["status"], "received")
        prepared = api.handle("POST", f"/sessions/{session_id}/knowledge/prepare", {"steward": "Arjun"}, "Meera")
        self.assertTrue(prepared["ok"], prepared)
        knowledge = prepared["data"]["knowledge"]
        self.assertTrue(knowledge["ready"])
        self.assertGreater(knowledge["vectorCount"], 0)
        self.assertGreater(knowledge["graphNodes"], 1)
        self.assertGreater(knowledge["graphEdges"], 0)
        self.assertTrue(prepared["data"]["evidenceReady"])

    def test_persona_routes_keep_the_campaign_name_and_update_open_sessions(self) -> None:
        templates = self.api.handle("GET", "/personas/templates", None, "Meera")
        self.assertTrue(templates["ok"], templates)
        self.assertGreaterEqual(len(templates["data"]), 1)

        before = self.api.handle("GET", "/campaigns/CAM-1842", None, "Meera")
        self.assertTrue(before["ok"], before)
        self.assertEqual(before["data"]["id"], "CAM-1842")
        self.assertIn("name", before["data"])
        self.assertEqual(before["data"]["persona"], "")

        opened = self.api.handle("POST", "/campaigns/CAM-1842/sessions", {}, "Meera")
        self.assertTrue(opened["ok"], opened)
        purpose_before = self.api.engine.store.read(opened["data"]["id"])["kit"]["purpose"]
        impact = self.api.handle("GET", "/campaigns/CAM-1842/impact", None, "Meera")
        self.assertTrue(impact["ok"], impact)
        self.assertGreaterEqual(impact["data"]["openSessions"], 1)

        empty = self.api.handle("PATCH", "/campaigns/CAM-1842/kit", {}, "Meera")
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["code"], "PACK_INVALID")

        persona = "VP of Sales at a large FMCG company. Speak from field execution, and ask before accepting a new system."
        saved = self.api.handle("PATCH", "/campaigns/CAM-1842/kit", {"persona": persona}, "Meera")
        self.assertTrue(saved["ok"], saved)
        self.assertEqual(saved["data"]["persona"], persona)
        self.assertGreaterEqual(saved["data"]["openSessions"], 1)

        after = self.api.handle("GET", "/campaigns/CAM-1842", None, "Meera")
        self.assertEqual(after["data"]["name"], before["data"]["name"])
        self.assertEqual(after["data"]["persona"], persona)

        session = self.api.engine.store.read(opened["data"]["id"])
        self.assertEqual(session["kit"]["persona"], persona)
        self.assertEqual(session["kit"]["purpose"], purpose_before)


if __name__ == "__main__":
    unittest.main()
