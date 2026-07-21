import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


radar = load_module("run_literature_radar", ROOT / "run_literature_radar.py")
mailer = load_module("send_email", ROOT / "send_email.py")


def base_config():
    return {
        "strong_keywords": ["self-driving lab"],
        "context_keywords": ["materials discovery"],
        "negative_keywords": [],
        "arxiv_categories": ["cs.AI", "cs.LG"],
        "llm": {"min_rule_score_for_review": 3},
    }


class CoreTests(unittest.TestCase):
    def test_qumus_style_embodied_ai_experimentalist_is_recalled(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        paper = radar.Paper(
            arxiv_id="2607.99999v1",
            title="Qumus: Realization of an Embodied AI Quantum Material Experimentalist",
            authors=[],
            summary=(
                "A multimodal multi-agent system performs closed-loop experimentation, "
                "autonomous error correction, and real-world scientific discovery."
            ),
            published="",
            updated="",
            categories=["cond-mat.mtrl-sci"],
            abs_url="",
            pdf_url="",
        )

        scored = radar.score_with_rules(paper, config)

        self.assertNotEqual(scored.decision, "exclude")

    def test_ai_experimentalist_phrase_is_a_strong_match(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        paper = radar.Paper(
            arxiv_id="2607.66666v1",
            title="An Embodied AI Experimentalist for Physics",
            authors=[],
            summary="A multimodal agent plans, operates, and diagnoses physical measurements.",
            published="",
            updated="",
            categories=["physics.app-ph"],
            abs_url="",
            pdf_url="",
        )

        scored = radar.score_with_rules(paper, config)

        self.assertEqual(scored.decision, "include")

    def test_generic_embodied_ai_robotics_is_not_promoted(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        paper = radar.Paper(
            arxiv_id="2607.88888v1",
            title="Embodied AI for Humanoid Locomotion",
            authors=[],
            summary="A physical AI policy improves walking and game-play control.",
            published="",
            updated="",
            categories=["cs.RO"],
            abs_url="",
            pdf_url="",
        )

        scored = radar.score_with_rules(paper, config)

        self.assertEqual(scored.decision, "exclude")

    def test_llm_prompt_requests_broad_ai_lab_and_rp_transfer_tags(self):
        paper = radar.Paper(
            arxiv_id="2607.77777v1",
            title="An AI Experimentalist",
            authors=[],
            summary="An autonomous physical experiment platform.",
            published="",
            updated="",
            categories=["physics.app-ph"],
            abs_url="",
            pdf_url="",
        )

        prompt = radar.build_classification_prompt(paper)

        self.assertIn("embodied AI", prompt)
        self.assertIn("rp_transfer", prompt)
        self.assertIn("Do not require photonics", prompt)

    def test_llm_response_parses_transfer_tags(self):
        response_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "include",
                                    "relevance": "strong",
                                    "rationale": "真实物理实验中的自主故障恢复方法。",
                                    "ai_lab_type": "embodied experimentalist",
                                    "domain": "quantum materials",
                                    "capabilities": ["fault recovery", "tool use"],
                                    "rp_transfer": ["state verification"],
                                    "priority": "high",
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return response_body

        with patch.object(radar.urllib.request, "urlopen", return_value=FakeResponse()):
            result = radar.call_chat_completion(
                "prompt", "secret", "model", {"base_url": "https://example.invalid"}
            )

        self.assertEqual(result["decision"], "include")
        self.assertEqual(result["rp_transfer"], ["state verification"])
        self.assertEqual(result["priority"], "high")

    def test_category_bonus_is_capped_and_cannot_enter_review_alone(self):
        paper = radar.Paper(
            arxiv_id="1234.56789v1",
            title="A generic AI paper",
            authors=[],
            summary="No topic keyword is present.",
            published="",
            updated="",
            categories=["cs.AI", "cs.LG"],
            abs_url="",
            pdf_url="",
        )

        scored = radar.score_with_rules(paper, base_config())

        self.assertEqual(scored.rule_score, 1)
        self.assertEqual(scored.decision, "exclude")


    def test_single_context_keyword_plus_category_stays_excluded(self):
        paper = radar.Paper(
            arxiv_id="1234.56789v1",
            title="Materials discovery with autonomous screening",
            authors=[],
            summary="This paper studies materials discovery.",
            published="",
            updated="",
            categories=["cs.AI"],
            abs_url="",
            pdf_url="",
        )

        scored = radar.score_with_rules(paper, base_config())

        self.assertEqual(scored.rule_score, 2)
        self.assertEqual(scored.decision, "exclude")

    def test_two_context_keywords_plus_category_reach_review_threshold(self):
        paper = radar.Paper(
            arxiv_id="1234.56789v1",
            title="Materials discovery with autonomous screening",
            authors=[],
            summary="This paper studies materials discovery.",
            published="",
            updated="",
            categories=["cs.AI"],
            abs_url="",
            pdf_url="",
        )
        config = base_config()
        config["context_keywords"] = ["materials discovery", "autonomous screening"]

        scored = radar.score_with_rules(paper, config)

        self.assertEqual(scored.rule_score, 3)
        self.assertEqual(scored.decision, "review")


    def test_secret_like_config_fields_are_rejected(self):
        config_path = Path("config.json")

        with self.assertRaisesRegex(RuntimeError, "GitHub Secrets"):
            radar.reject_secrets_in_config({"llm": {"api_key": "do-not-store"}}, config_path)


    def test_mail_to_is_required(self):
        old_value = os.environ.pop("MAIL_TO", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "MAIL_TO"):
                mailer.require_env("MAIL_TO")
        finally:
            if old_value is not None:
                os.environ["MAIL_TO"] = old_value

    def test_state_round_trip(self):
        state_dir = ROOT / ".test-state"
        shutil.rmtree(state_dir, ignore_errors=True)
        state_dir.mkdir(parents=True)
        try:
            radar.save_seen_ids(state_dir, {"2607.00002v1", "2607.00001v1"})

            self.assertEqual(
                radar.load_seen_ids(state_dir),
                {"2607.00001v1", "2607.00002v1"},
            )
            stored = json.loads((state_dir / "seen_arxiv_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, ["2607.00001v1", "2607.00002v1"])
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_atom_xml_page_parsing(self):
        atom = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2607.00001v1</id>
            <updated>2026-07-15T00:00:00Z</updated>
            <published>2026-07-14T23:00:00Z</published>
            <title>  Agentic Self-Driving Lab for Chemistry  </title>
            <summary>
              A closed-loop experiment platform for automated chemistry.
            </summary>
            <author><name>Ada Lovelace</name></author>
            <category term="cs.AI"/>
            <category term="physics.chem-ph"/>
            <link title="pdf" href="https://arxiv.org/pdf/2607.00001v1" type="application/pdf"/>
          </entry>
        </feed>
        """

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return atom

        with patch.object(radar.urllib.request, "urlopen", return_value=FakeResponse()):
            papers = radar.fetch_arxiv_page("all:test", 0, 1, 1, 0, 0)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2607.00001v1")
        self.assertEqual(papers[0].title, "Agentic Self-Driving Lab for Chemistry")
        self.assertEqual(papers[0].authors, ["Ada Lovelace"])
        self.assertEqual(papers[0].categories, ["cs.AI", "physics.chem-ph"])
        self.assertEqual(papers[0].pdf_url, "https://arxiv.org/pdf/2607.00001v1")

    def test_report_rendering_contains_sections_and_no_match_message(self):
        config = {"topic_name": "Test Radar", "lookback_days": 14}
        start = radar.datetime(2026, 7, 1, tzinfo=radar.timezone.utc)
        end = radar.datetime(2026, 7, 15, tzinfo=radar.timezone.utc)
        paper = radar.Paper(
            arxiv_id="2607.00001v1",
            title="Agentic Self-Driving Lab",
            authors=["Ada Lovelace"],
            summary="",
            published="2026-07-14T23:00:00Z",
            updated="",
            categories=["cs.AI"],
            abs_url="http://arxiv.org/abs/2607.00001v1",
            pdf_url="https://arxiv.org/pdf/2607.00001v1",
            decision="include",
            relevance="strong",
            rationale="Relevant.",
            rule_reasons=["strong: self-driving lab"],
        )

        report = radar.render_markdown([paper], config, start, end)
        self.assertIn("## Strong Matches", report)
        self.assertIn("Agentic Self-Driving Lab", report)
        self.assertIn("Lookback days: `14`", report)

        empty_report = radar.render_markdown([], config, start, end)
        self.assertIn("## No Relevant New Papers", empty_report)

    def test_email_rendering_is_chinese_html_and_omits_run_log(self):
        config = {"topic_name": "Test Radar", "lookback_days": 14}
        start = radar.datetime(2026, 7, 1, tzinfo=radar.timezone.utc)
        end = radar.datetime(2026, 7, 15, tzinfo=radar.timezone.utc)
        paper = radar.Paper(
            arxiv_id="2607.00001v1",
            title="Agentic Self-Driving Lab",
            authors=["Ada Lovelace"],
            summary="A platform for closed-loop automated chemistry experiments.",
            published="2026-07-14T23:00:00Z",
            updated="",
            categories=["cs.AI"],
            abs_url="http://arxiv.org/abs/2607.00001v1",
            pdf_url="https://arxiv.org/pdf/2607.00001v1",
            decision="include",
            relevance="strong",
            rationale="这篇文章介绍了面向自动化化学实验的闭环平台。",
            rule_reasons=["strong: self-driving lab"],
            rp_transfer=["workflow hierarchy", "fault recovery"],
        )

        html = radar.render_email_html([paper], config, start, end)
        text = radar.render_email_text([paper], config, start, end)

        self.assertIn("每周 arXiv 自动化实验室文献雷达", html)
        self.assertIn("中文简介", html)
        self.assertIn("为什么值得看", text)
        self.assertIn("workflow hierarchy", html)
        self.assertIn("fault recovery", text)
        self.assertNotIn("Fetched:", html)
        self.assertNotIn("Wrote ", text)


if __name__ == "__main__":
    unittest.main()
