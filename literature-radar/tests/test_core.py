import importlib.util
import io
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
    def test_arxiv_queries_are_split_into_small_term_batches(self):
        config = base_config()
        config["strong_keywords"] = [f"strong-{index}" for index in range(7)]
        config["context_keywords"] = ["context-1", "context-2"]
        config["query_terms_per_request"] = 3
        start = radar.datetime(2026, 7, 1, tzinfo=radar.timezone.utc)
        end = radar.datetime(2026, 7, 15, tzinfo=radar.timezone.utc)

        queries = radar.build_queries(config, start, end)

        self.assertEqual(len(queries), 3)
        self.assertEqual([query.count('all:"') for query in queries], [3, 3, 3])
        self.assertIn('all:"strong-0"', queries[0])
        self.assertIn('all:"context-2"', queries[2])
        self.assertTrue(all("submittedDate:[202607010000 TO 202607150000]" in query for query in queries))

    def test_openalex_searches_use_the_same_small_term_batches(self):
        config = base_config()
        config["strong_keywords"] = [f"strong-{index}" for index in range(5)]
        config["context_keywords"] = ["context-1", "context-2"]
        config["query_terms_per_request"] = 3

        searches = radar.build_openalex_searches(config)

        self.assertEqual(len(searches), 3)
        self.assertEqual([search.count('"') for search in searches], [6, 6, 2])
        self.assertEqual(searches[0], '("strong-0" OR "strong-1" OR "strong-2")')
        self.assertEqual(searches[2], '("context-2")')

    def test_arxiv_query_batches_are_deduplicated_sorted_and_capped(self):
        def paper(arxiv_id, published):
            return radar.Paper(
                arxiv_id=arxiv_id,
                title=arxiv_id,
                authors=[],
                summary="",
                published=published,
                updated=published,
                categories=[],
                abs_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )

        older = paper("2607.00001v1", "2026-07-10T00:00:00Z")
        duplicate = paper("2607.00002v1", "2026-07-11T00:00:00Z")
        newer = paper("2607.00003v1", "2026-07-12T00:00:00Z")
        with (
            patch.object(radar, "fetch_arxiv", side_effect=[[older, duplicate], [newer, duplicate]]) as fetch_mock,
            patch.object(radar.time, "sleep"),
        ):
            papers = radar.fetch_arxiv_queries(
                ["query-1", "query-2"],
                max_results=2,
                max_results_per_query=30,
                page_size=30,
                retries=2,
                retry_initial_delay=1,
                retry_max_delay=2,
                retry_total_budget=None,
                request_interval=0,
            )

        self.assertEqual([item.arxiv_id for item in papers], ["2607.00003v1", "2607.00002v1"])
        self.assertEqual(fetch_mock.call_count, 2)

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
                {"2607.00001", "2607.00002"},
            )
            stored = json.loads((state_dir / "seen_arxiv_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, ["2607.00001", "2607.00002"])
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
        self.assertEqual(papers[0].arxiv_id, "2607.00001")
        self.assertEqual(papers[0].title, "Agentic Self-Driving Lab for Chemistry")
        self.assertEqual(papers[0].authors, ["Ada Lovelace"])
        self.assertEqual(papers[0].categories, ["cs.AI", "physics.chem-ph"])
        self.assertEqual(papers[0].pdf_url, "https://arxiv.org/pdf/2607.00001v1")

    def test_openalex_work_reconstructs_abstract_and_doi_arxiv_location(self):
        work = {
            "display_name": "Agentic Self-Driving Lab",
            "publication_date": "2026-07-14",
            "updated_date": "2026-07-15T12:00:00Z",
            "abstract_inverted_index": {
                "closed-loop": [1],
                "A": [0],
                "platform": [2],
            },
            "authorships": [
                {"author": {"display_name": "Ada Lovelace"}},
                {"author": {"display_name": "Grace Hopper"}},
            ],
            "locations": [
                {
                    "source": {"id": "https://openalex.org/S4306400194"},
                    "landing_page_url": "https://doi.org/10.48550/arXiv.2607.00001v2",
                    "pdf_url": None,
                }
            ],
        }

        paper = radar.paper_from_openalex_work(work)

        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.arxiv_id, "2607.00001")
        self.assertEqual(paper.summary, "A closed-loop platform")
        self.assertEqual(paper.authors, ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(paper.published, "2026-07-14T00:00:00Z")
        self.assertEqual(paper.abs_url, "https://arxiv.org/abs/2607.00001")
        self.assertEqual(paper.pdf_url, "https://arxiv.org/pdf/2607.00001")

    def test_openalex_batches_are_deduplicated_and_partial_failure_is_tolerated(self):
        config = base_config()
        config["context_keywords"] = ["materials discovery", "automated experimentation"]
        config.update(
            {
                "max_results": 10,
                "max_results_per_query": 10,
                "query_terms_per_request": 1,
                "openalex_request_interval_seconds": 0,
            }
        )
        paper = radar.Paper(
            arxiv_id="2607.00001",
            title="Agentic Self-Driving Lab",
            authors=[],
            summary="",
            published="2026-07-14T00:00:00Z",
            updated="",
            categories=[],
            abs_url="https://arxiv.org/abs/2607.00001",
            pdf_url="https://arxiv.org/pdf/2607.00001",
        )
        old_paper = radar.Paper(
            arxiv_id="2401.03428",
            title="Old arXiv preprint with a new journal date",
            authors=[],
            summary="",
            published="2026-07-14T00:00:00Z",
            updated="",
            categories=[],
            abs_url="https://arxiv.org/abs/2401.03428",
            pdf_url="https://arxiv.org/pdf/2401.03428",
        )
        start = radar.datetime(2026, 7, 1, tzinfo=radar.timezone.utc)
        end = radar.datetime(2026, 7, 15, tzinfo=radar.timezone.utc)
        with (
            patch.object(
                radar,
                "fetch_openalex_search",
                side_effect=[[paper, old_paper], RuntimeError("temporary"), [paper]],
            ),
            patch.object(radar.time, "sleep"),
            patch.object(radar.sys, "stderr", io.StringIO()),
        ):
            papers = radar.fetch_openalex(config, start, end)

        self.assertEqual([item.arxiv_id for item in papers], ["2607.00001"])

    def test_arxiv_failure_falls_back_to_openalex(self):
        config = base_config()
        config.update(
            {
                "max_results": 10,
                "max_results_per_query": 10,
                "page_size": 10,
                "query_terms_per_request": 10,
            }
        )
        fallback_paper = radar.Paper(
            arxiv_id="2607.00001",
            title="Fallback result",
            authors=[],
            summary="",
            published="2026-07-14T00:00:00Z",
            updated="",
            categories=[],
            abs_url="https://arxiv.org/abs/2607.00001",
            pdf_url="https://arxiv.org/pdf/2607.00001",
        )
        start = radar.datetime(2026, 7, 1, tzinfo=radar.timezone.utc)
        end = radar.datetime(2026, 7, 15, tzinfo=radar.timezone.utc)
        with (
            patch.object(radar, "fetch_arxiv_queries", side_effect=RuntimeError("rate limited")),
            patch.object(radar, "fetch_openalex", return_value=[fallback_paper]) as fallback_mock,
            patch.object(radar.sys, "stderr", io.StringIO()),
        ):
            papers, source, batches = radar.fetch_papers_with_fallback(config, start, end)

        self.assertEqual(papers, [fallback_paper])
        self.assertEqual(source, "OpenAlex arXiv index")
        self.assertEqual(batches, 1)
        fallback_mock.assert_called_once_with(config, start, end)

    def test_arxiv_429_honors_retry_after_then_recovers(self):
        atom = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>
        """

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return atom

        rate_limited = radar.urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            429,
            "Too Many Requests",
            {"Retry-After": "7"},
            None,
        )
        with (
            patch.object(radar.urllib.request, "urlopen", side_effect=[rate_limited, FakeResponse()]) as open_mock,
            patch.object(radar.random, "uniform", return_value=0),
            patch.object(radar.time, "sleep") as sleep_mock,
            patch.object(radar.sys, "stderr", io.StringIO()),
        ):
            papers = radar.fetch_arxiv_page("all:test", 0, 1, 2, 1, 60)

        self.assertEqual(papers, [])
        self.assertEqual(open_mock.call_count, 2)
        sleep_mock.assert_called_once_with(7.0)

    def test_arxiv_permanent_http_error_is_not_retried(self):
        bad_request = radar.urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            400,
            "Bad Request",
            {},
            None,
        )
        with (
            patch.object(radar.urllib.request, "urlopen", side_effect=bad_request) as open_mock,
            patch.object(radar.time, "sleep") as sleep_mock,
        ):
            with self.assertRaises(radar.urllib.error.HTTPError):
                radar.fetch_arxiv_page("all:test", 0, 1, 4, 1, 60)

        self.assertEqual(open_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_arxiv_transient_error_without_headers_uses_backoff(self):
        atom = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return atom

        unavailable = radar.urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            503,
            "Service Unavailable",
            None,
            None,
        )
        with (
            patch.object(radar.urllib.request, "urlopen", side_effect=[unavailable, FakeResponse()]),
            patch.object(radar.random, "uniform", return_value=0),
            patch.object(radar.time, "sleep") as sleep_mock,
            patch.object(radar.sys, "stderr", io.StringIO()),
        ):
            papers = radar.fetch_arxiv_page("all:test", 0, 1, 2, 2, 60)

        self.assertEqual(papers, [])
        sleep_mock.assert_called_once_with(2)

    def test_arxiv_retry_stops_when_total_budget_is_insufficient(self):
        rate_limited = radar.urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            429,
            "Too Many Requests",
            {"Retry-After": "60"},
            None,
        )
        with (
            patch.object(radar.urllib.request, "urlopen", side_effect=rate_limited) as open_mock,
            patch.object(radar.random, "uniform", return_value=0),
            patch.object(radar.time, "monotonic", return_value=100),
            patch.object(radar.time, "sleep") as sleep_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "retry budget exhausted"):
                radar.fetch_arxiv_page(
                    "all:test",
                    0,
                    1,
                    4,
                    1,
                    60,
                    retry_deadline=110,
                )

        self.assertEqual(open_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_retry_after_http_date_is_parsed(self):
        now = radar.datetime(2026, 7, 27, 0, 0, tzinfo=radar.timezone.utc)
        delay = radar.parse_retry_after("Mon, 27 Jul 2026 00:00:30 GMT", now)
        self.assertEqual(delay, 30)

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
