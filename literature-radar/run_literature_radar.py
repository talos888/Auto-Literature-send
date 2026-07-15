#!/usr/bin/env python3
"""Weekly arXiv literature radar for AI-driven / automated laboratories.

The script is dependency-free by default. If DEEPSEEK_API_KEY is available and
config.json enables LLM review, borderline/relevant candidates are classified
with DeepSeek's OpenAI-compatible chat API. Otherwise it falls back to
transparent rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
SECRET_FIELD_NAMES = {"api_key", "apikey", "secret", "token", "password"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    categories: list[str]
    abs_url: str
    pdf_url: str
    rule_score: int = 0
    rule_reasons: list[str] | None = None
    decision: str = "unknown"
    relevance: str = "unknown"
    rationale: str = ""


def utc_now() -> datetime:
    override = os.environ.get("RADAR_NOW_UTC")
    if override:
        return datetime.fromisoformat(override.replace("Z", "+00:00")).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    reject_secrets_in_config(config, path)
    return config


def reject_secrets_in_config(value: Any, path: Path, trail: str = "") -> None:
    """Fail closed if someone tries to store credentials in JSON config."""
    if isinstance(value, dict):
        for key, nested in value.items():
            next_trail = f"{trail}.{key}" if trail else key
            if key.lower() in SECRET_FIELD_NAMES:
                raise RuntimeError(
                    f"Refusing to read secret-like field `{next_trail}` from {path}. "
                    "Store credentials only in GitHub Secrets or environment variables."
                )
            reject_secrets_in_config(nested, path, next_trail)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_secrets_in_config(nested, path, f"{trail}[{index}]")


def normalize_arxiv_id(abs_url: str) -> str:
    return abs_url.rstrip("/").split("/")[-1]


def build_query(config: dict[str, Any], start: datetime, end: datetime) -> str:
    date_filter = f"submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    all_terms = config["strong_keywords"] + config["context_keywords"]
    keyword_terms = " OR ".join(f'all:"{term}"' for term in all_terms)
    return f"({date_filter}) AND ({keyword_terms})"


def fetch_arxiv(
    query: str,
    max_results: int,
    page_size: int,
    retries: int,
    retry_initial_delay: float,
    retry_max_delay: float,
) -> list[Paper]:
    papers: list[Paper] = []
    seen_ids: set[str] = set()
    page_size = max(1, min(page_size, 100))
    for start in range(0, max_results, page_size):
        batch_size = min(page_size, max_results - start)
        batch = fetch_arxiv_page(query, start, batch_size, retries, retry_initial_delay, retry_max_delay)
        if not batch:
            break
        for paper in batch:
            if paper.arxiv_id not in seen_ids:
                papers.append(paper)
                seen_ids.add(paper.arxiv_id)
        if len(batch) < batch_size:
            break
        time.sleep(3.1)
    return papers


def fetch_arxiv_page(
    query: str,
    start: int,
    max_results: int,
    retries: int,
    retry_initial_delay: float,
    retry_max_delay: float,
) -> list[Paper]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "literature-radar/0.1"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                root = ET.fromstring(response.read())
            break
        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(retry_max_delay, retry_initial_delay * (2 ** (attempt - 1)))
            time.sleep(delay)

    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM_NS):
        abs_url = entry.findtext("a:id", default="", namespaces=ATOM_NS)
        links = entry.findall("a:link", ATOM_NS)
        pdf_url = ""
        for link in links:
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        papers.append(
            Paper(
                arxiv_id=normalize_arxiv_id(abs_url),
                title=clean_text(entry.findtext("a:title", default="", namespaces=ATOM_NS)),
                authors=[
                    clean_text(author.findtext("a:name", default="", namespaces=ATOM_NS))
                    for author in entry.findall("a:author", ATOM_NS)
                ],
                summary=clean_text(entry.findtext("a:summary", default="", namespaces=ATOM_NS)),
                published=entry.findtext("a:published", default="", namespaces=ATOM_NS),
                updated=entry.findtext("a:updated", default="", namespaces=ATOM_NS),
                categories=[cat.attrib.get("term", "") for cat in entry.findall("a:category", ATOM_NS)],
                abs_url=abs_url,
                pdf_url=pdf_url or abs_url.replace("/abs/", "/pdf/"),
            )
        )
    return papers


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def keyword_hits(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def score_with_rules(paper: Paper, config: dict[str, Any]) -> Paper:
    text = f"{paper.title}\n{paper.summary}"
    strong_hits = keyword_hits(text, config["strong_keywords"])
    context_hits = keyword_hits(text, config["context_keywords"])
    negative_hits = keyword_hits(text, config["negative_keywords"])
    category_hits = sorted(set(paper.categories).intersection(config.get("arxiv_categories", [])))
    category_bonus = 1 if category_hits else 0

    score = 3 * len(strong_hits) + len(context_hits) + category_bonus - 2 * len(negative_hits)
    reasons: list[str] = []
    if strong_hits:
        reasons.append("strong: " + ", ".join(strong_hits[:5]))
    if context_hits:
        reasons.append("context: " + ", ".join(context_hits[:5]))
    if negative_hits:
        reasons.append("negative: " + ", ".join(negative_hits[:5]))
    if category_hits:
        reasons.append("preferred category: " + ", ".join(category_hits[:5]))

    paper.rule_score = score
    paper.rule_reasons = reasons
    if score >= 4:
        paper.decision = "include"
        paper.relevance = "strong"
        paper.rationale = "Rule score indicates a close match to AI-driven or automated laboratory work."
    elif score >= review_threshold(config):
        paper.decision = "review"
        paper.relevance = "weak"
        paper.rationale = "Rule score suggests possible relevance; manual or LLM review is useful."
    else:
        paper.decision = "exclude"
        paper.relevance = "unlikely"
        paper.rationale = "No sufficient topic evidence from title/abstract keywords."
    return paper


def classify_with_llm(papers: list[Paper], config: dict[str, Any]) -> None:
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "deepseek")
    api_key = llm_api_key(provider)
    if not api_key:
        return

    model = llm_config.get("model", "deepseek-v4-flash")
    for paper in papers:
        if paper.rule_score < review_threshold(config):
            continue
        prompt = build_classification_prompt(paper)
        try:
            result = call_chat_completion(prompt, api_key, model, llm_config)
            paper.decision = result["decision"]
            paper.relevance = result["relevance"]
            paper.rationale = result["rationale"]
            time.sleep(float(llm_config.get("request_delay_seconds", 2)))
        except Exception as exc:  # Keep the scheduled job useful even if LLM is down.
            paper.rationale = f"{paper.rationale} LLM review failed: {redact_secret(str(exc), api_key)}"


def llm_api_key(provider: str) -> str | None:
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    env_name = f"{provider.upper()}_API_KEY"
    return os.environ.get(env_name)


def redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED_API_KEY]")


def build_classification_prompt(paper: Paper) -> str:
    return textwrap.dedent(
        f"""
        You are screening new scientific papers for a weekly alert on AI-driven
        automated laboratories and self-driving scientific discovery.

        Include papers about real or proposed scientific-experiment workflows
        involving AI agents, machine learning, optimization, robotics, lab
        automation, closed-loop experimentation, autonomous chemistry/materials
        platforms, or automated wet-lab/physical experiments.

        Exclude papers that only use closed-loop/control language in unrelated
        domains such as blockchain, networking, generic robotics without a lab
        or scientific discovery component, or generic AI without experiments.

        Return JSON only.

        Title: {paper.title}
        Categories: {", ".join(paper.categories)}
        Abstract: {paper.summary}
        Rule score: {paper.rule_score}
        Rule reasons: {"; ".join(paper.rule_reasons or [])}
        """
    ).strip()


def call_chat_completion(
    prompt: str,
    api_key: str,
    model: str,
    llm_config: dict[str, Any],
) -> dict[str, str]:
    base_url = llm_config.get("base_url", "https://api.deepseek.com").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify scientific papers for a weekly literature radar. "
                    "Return only valid JSON with keys decision, relevance, rationale."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = json.loads(response.read().decode("utf-8"))

    content = raw["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if parsed.get("decision") not in {"include", "review", "exclude"}:
        raise RuntimeError(f"Invalid decision: {parsed.get('decision')}")
    if parsed.get("relevance") not in {"strong", "weak", "unlikely"}:
        raise RuntimeError(f"Invalid relevance: {parsed.get('relevance')}")
    return {
        "decision": parsed["decision"],
        "relevance": parsed["relevance"],
        "rationale": str(parsed.get("rationale", "")),
    }


def review_threshold(config: dict[str, Any]) -> int:
    return int(config.get("llm", {}).get("min_rule_score_for_review", 3))


def render_markdown(papers: list[Paper], config: dict[str, Any], start: datetime, end: datetime) -> str:
    included = [p for p in papers if p.decision == "include"]
    review = [p for p in papers if p.decision == "review"]
    excluded_count = sum(1 for p in papers if p.decision == "exclude")

    lines = [
        f"# Weekly Literature Radar: {config['topic_name']}",
        "",
        f"- Window UTC: `{start.isoformat()}` to `{end.isoformat()}`",
        f"- Total arXiv candidates scanned: `{len(papers)}`",
        f"- Lookback days: `{config['lookback_days']}`",
        f"- Included: `{len(included)}`",
        f"- Review: `{len(review)}`",
        f"- Excluded: `{excluded_count}`",
        "",
    ]
    lines.extend(render_section("Strong Matches", included))
    lines.extend(render_section("Review Queue", review))
    if not included and not review:
        lines.extend(["## No Relevant New Papers", "", "No papers passed the current rule/LLM screen this week.", ""])
    return "\n".join(lines)


def render_section(title: str, papers: list[Paper]) -> list[str]:
    lines = [f"## {title}", ""]
    if not papers:
        lines.extend(["None.", ""])
        return lines
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:6])
        if len(paper.authors) > 6:
            authors += ", et al."
        lines.extend(
            [
                f"### {index}. {paper.title}",
                "",
                f"- Authors: {authors}",
                f"- Published: `{paper.published}`",
                f"- Categories: `{', '.join(paper.categories)}`",
                f"- Relevance: `{paper.relevance}`",
                f"- Reason: {paper.rationale}",
                f"- Rule evidence: {'; '.join(paper.rule_reasons or ['none'])}",
                f"- arXiv: {paper.abs_url}",
                f"- PDF: {paper.pdf_url}",
                "",
            ]
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="literature-radar/config.json")
    parser.add_argument("--out-dir", default="literature-radar/out")
    parser.add_argument("--state-dir", default="literature-radar/.radar-state")
    args = parser.parse_args()

    if any("api" in arg.lower() and "key" in arg.lower() for arg in sys.argv[1:]):
        raise RuntimeError("Do not pass API keys on the command line. Use environment variables only.")

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    state_dir = Path(args.state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    end = utc_now()
    start = end - timedelta(days=int(config["lookback_days"]))
    query = build_query(config, start, end)

    seen_ids = load_seen_ids(state_dir)
    fetched_papers = fetch_arxiv(
        query,
        int(config["max_results"]),
        int(config.get("page_size", 100)),
        int(config.get("request_retries", 3)),
        float(config.get("retry_initial_delay_seconds", 5)),
        float(config.get("retry_max_delay_seconds", 60)),
    )
    papers = [paper for paper in fetched_papers if paper.arxiv_id not in seen_ids]
    papers = [score_with_rules(paper, config) for paper in papers]
    if config.get("llm", {}).get("enabled", False):
        classify_with_llm(papers, config)

    papers.sort(key=lambda p: (p.decision != "include", p.decision != "review", -p.rule_score, p.published))
    report = render_markdown(papers, config, start, end)

    stamp = end.strftime("%Y-%m-%d")
    json_path = out_dir / f"literature-radar-{stamp}.json"
    md_path = out_dir / f"literature-radar-{stamp}.md"
    json_path.write_text(json.dumps([asdict(p) for p in papers], ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report, encoding="utf-8")
    save_seen_ids(state_dir, seen_ids.union({paper.arxiv_id for paper in fetched_papers}))

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Fetched: {len(fetched_papers)}")
    print(f"New candidates: {len(papers)}")
    print(f"Included: {sum(1 for p in papers if p.decision == 'include')}")
    print(f"Review: {sum(1 for p in papers if p.decision == 'review')}")
    return 0


def load_seen_ids(state_dir: Path) -> set[str]:
    state_path = state_dir / "seen_arxiv_ids.json"
    if not state_path.exists():
        return set()
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid state file format: {state_path}")
    return {str(item) for item in data}


def save_seen_ids(state_dir: Path, seen_ids: set[str]) -> None:
    state_path = state_dir / "seen_arxiv_ids.json"
    state_path.write_text(json.dumps(sorted(seen_ids), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
