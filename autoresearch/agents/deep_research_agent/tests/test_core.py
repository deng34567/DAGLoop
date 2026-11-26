"""
Tests for DeepResearchAgent's pure two-pass-search + synthesis core.
DeepResearchAgent 纯"两轮搜索 + 综述合成"核心的测试。

No EventBus / Store / Redis / network — only a FakeBackend (canned Paper
lists) + a FakeLLM (canned JSON queue).
不涉及 EventBus / Store / Redis / 网络 —— 只用 FakeBackend（预设 Paper 列表）+
FakeLLM（预设 JSON 响应队列）。

(c) is the key anti-hallucination test: the LLM cites a paper that was never
retrieved by either search pass, and the EXPLICIT parse-time grounding check
in core.py must reject it (never trust the prompt alone).
(c) 是关键的抗幻觉测试：LLM 引用了一篇两轮搜索都未检索到的论文，core.py 中
显式的解析期溯源校验必须拒绝它（绝不仅依赖 prompt）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.deep_research_backend import FakeBackend
from autoresearch.agents.deep_research_agent.core import (
    MAX_RETRIES,
    DeepResearchParseError,
    DeepResearchResult,
    _build_user_message,
    _invalid_json_reason,
    _load_system_prompt,
    deep_research_step,
)
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


LITERATURE_BRIEF_VALIDATOR = _validator("literature_brief.schema.yaml")
FIELD_TRAJECTORY_BRIEF_VALIDATOR = _validator("field_trajectory_brief.schema.yaml")

RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Survey RAG methods for long-tail scientific question answering.",
}

SEMINAL_PAPERS = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "id_or_url": "arxiv:2005.11401",
        "year": 2020,
        "note": "Seminal RAG paper.",
    },
]

FRONTIER_PAPERS = [
    {
        "title": "Long-Tail Scientific QA via Retrieval-Augmented Fine-Tuning",
        "id_or_url": "arxiv:2401.00000",
        "year": 2024,
        "note": "Recent frontier work on long-tail scientific QA.",
    },
]

VALID_LITERATURE_BRIEF = {
    "summary": "The field combines dense retrieval with generative models to ground answers.",
    "related_work": [],
    "key_papers": [SEMINAL_PAPERS[0], FRONTIER_PAPERS[0]],
    "existing_benchmarks": [
        {"name": "Natural Questions", "note": "Standard open-domain QA benchmark."},
    ],
}

VALID_FIELD_TRAJECTORY_BRIEF = {
    "origin": "RAG originated from combining dense retrieval with seq2seq generation.",
    "current_state": "Frontier work extends RAG to long-tail, low-resource scientific QA.",
    "main_method_families": ["dense retrieval", "retrieval-augmented fine-tuning"],
    "open_problems": ["long-tail coverage", "hallucination under sparse retrieval"],
}


def _synth_json(literature_brief: dict, field_trajectory_brief: dict) -> str:
    return json.dumps(
        {
            "literature_brief": literature_brief,
            "field_trajectory_brief": field_trajectory_brief,
        }
    )


# EVERY topic (English included) is now distilled to English keywords via one LLM call
# BEFORE retrieval, so a test that drives deep_research_step queues this keyword-planning
# response ahead of its synthesis response(s).
_PLAN_JSON = json.dumps({"query": "retrieval augmented generation scientific QA"})


def _retrieved_keys(*paper_lists: list[dict]) -> set[str]:
    keys: set[str] = set()
    for papers in paper_lists:
        for paper in papers:
            keys.add(paper.get("id_or_url") or paper.get("title"))
    return keys


# ---------------------------------------------------------------------------
# (a) valid path -> DeepResearchResult with both briefs, schema-valid,
#     key_papers subset of retrieved set.
# ---------------------------------------------------------------------------


def test_valid_path_returns_schema_valid_grounded_briefs():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    llm = FakeLLM([_PLAN_JSON, _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF)])

    result = deep_research_step(RESEARCH_SPEC, backend, llm)

    assert isinstance(result, DeepResearchResult)

    lb_errors = list(LITERATURE_BRIEF_VALIDATOR.iter_errors(result.literature_brief))
    assert lb_errors == [], [e.message for e in lb_errors]

    ftb_errors = list(
        FIELD_TRAJECTORY_BRIEF_VALIDATOR.iter_errors(result.field_trajectory_brief)
    )
    assert ftb_errors == [], [e.message for e in ftb_errors]

    retrieved = _retrieved_keys(SEMINAL_PAPERS, FRONTIER_PAPERS)
    for paper in result.literature_brief["key_papers"]:
        key = paper.get("id_or_url") or paper.get("title")
        assert key in retrieved


# ---------------------------------------------------------------------------
# (b) both search passes invoked exactly once.
# ---------------------------------------------------------------------------


def test_both_search_passes_invoked_once():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    llm = FakeLLM([_PLAN_JSON, _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF)])

    deep_research_step(RESEARCH_SPEC, backend, llm)

    assert backend.seminal_calls == 1
    assert backend.frontier_calls == 1


def test_unknown_optional_year_retry_identifies_field_and_preserves_grounding():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    invalid = dict(VALID_LITERATURE_BRIEF, key_papers=[dict(SEMINAL_PAPERS[0], year=None)])
    corrected = dict(VALID_LITERATURE_BRIEF, key_papers=[
        {key: value for key, value in SEMINAL_PAPERS[0].items() if key != "year"}])
    llm = FakeLLM([_PLAN_JSON, _synth_json(invalid, VALID_FIELD_TRAJECTORY_BRIEF),
                   _synth_json(corrected, VALID_FIELD_TRAJECTORY_BRIEF)])
    result = deep_research_step(RESEARCH_SPEC, backend, llm)
    assert "year" not in result.literature_brief["key_papers"][0]
    assert result.literature_brief["key_papers"][0]["id_or_url"] == SEMINAL_PAPERS[0]["id_or_url"]
    assert any("key_papers.0.year: None is not of type" in message["content"]
               for call in llm.calls for message in call)


def test_multiple_directions_share_papers_and_claims_in_one_grounded_dag():
    shared = {
        "title": "Shared Evidence Paper", "id_or_url": "https://example.org/shared",
    }
    direction_a = {
        "title": "Direction A Paper", "id_or_url": "https://example.org/a",
    }
    direction_b = {
        "title": "Direction B Paper", "id_or_url": "https://example.org/b",
    }

    class Backend:
        def search_seminal(self, query, spec):
            return [shared, direction_a] if "retrieval" in query else [shared, direction_b]

        def search_frontier(self, query, spec):
            return []

        def fetch(self, papers):
            return [{**paper, "fetched_excerpt": "grounded full-text evidence"} for paper in papers]

    directions = {"directions": [
        {"id": "d1", "label": "retrieval", "query": "retrieval evidence",
         "basis": "clarification asks how evidence is found"},
        {"id": "d2", "label": "evaluation", "query": "evaluation evidence",
         "basis": "clarification asks how value is judged"},
    ]}
    papers = [shared, direction_a, direction_b]
    brief = {
        "summary": "Two directions share one evidence source and one conclusion.",
        "related_work": [], "key_papers": papers, "existing_benchmarks": [],
        "evidence_dag": {
            "directions": [
                {"id": "d1", "label": "retrieval", "basis": directions["directions"][0]["basis"],
                 "source_refs": ["clarify:evidence"]},
                {"id": "d2", "label": "evaluation", "basis": directions["directions"][1]["basis"],
                 "source_refs": ["clarify:value"]},
            ],
            "papers": [
                {"id": "p1", **shared, "authors": [], "year": None,
                 "evidence": [{"location": shared["id_or_url"],
                               "snippet": "grounded full-text evidence"}]},
                {"id": "p2", **direction_a, "authors": [], "year": None,
                 "evidence": [{"location": direction_a["id_or_url"],
                               "snippet": "grounded full-text evidence"}]},
                {"id": "p3", **direction_b, "authors": [], "year": None,
                 "evidence": [{"location": direction_b["id_or_url"],
                               "snippet": "grounded full-text evidence"}]},
            ],
            "claims": [{"id": "c1", "text": "The shared conclusion", "evidence": [
                {"paper_id": "p1", "location": shared["id_or_url"],
                 "snippet": "grounded full-text evidence"},
                {"paper_id": "p2", "location": direction_a["id_or_url"],
                 "snippet": "grounded full-text evidence"},
            ]}],
            "edges": [
                {"source": "d1", "target": "p1", "relation": "direction_has_paper"},
                {"source": "d2", "target": "p1", "relation": "direction_has_paper"},
                {"source": "d1", "target": "p2", "relation": "direction_has_paper"},
                {"source": "d2", "target": "p3", "relation": "direction_has_paper"},
                {"source": "p1", "target": "c1", "relation": "paper_supports_claim"},
                {"source": "p2", "target": "c1", "relation": "paper_supports_claim"},
            ],
        },
        "evidence_gaps": [],
    }
    llm = FakeLLM([
        json.dumps(directions), _synth_json(brief, VALID_FIELD_TRAJECTORY_BRIEF),
    ])

    result = deep_research_step(RESEARCH_SPEC, Backend(), llm)

    dag = result.literature_brief["evidence_dag"]
    assert len(dag["directions"]) == 2
    assert len([edge for edge in dag["edges"] if edge["target"] == "p1"]) == 2
    assert len([edge for edge in dag["edges"] if edge["target"] == "c1"]) == 2


def test_missing_direction_runs_one_bounded_gap_search_then_rebuilds_dag():
    p1 = {"title": "Alpha Paper", "id_or_url": "https://example.org/alpha"}
    p2 = {"title": "Beta Paper", "id_or_url": "https://example.org/beta"}

    class Backend:
        def __init__(self):
            self.beta_seminal_calls = 0

        def search_seminal(self, query, spec):
            if "alpha" in query:
                return [p1]
            if "beta" in query:
                self.beta_seminal_calls += 1
                return [] if self.beta_seminal_calls == 1 else [p2]
            return []

        def search_frontier(self, query, spec):
            return []

        def fetch(self, papers):
            return [
                {**paper, "fetched_excerpt": (
                    "alpha grounded evidence" if "alpha" in paper["id_or_url"]
                    else "beta grounded evidence")}
                for paper in papers
            ]

    directions = {"directions": [
        {"id": "d1", "label": "alpha", "query": "alpha mechanism",
         "basis": "user asks about alpha"},
        {"id": "d2", "label": "beta", "query": "beta mechanism",
         "basis": "clarification asks about beta"},
    ]}

    def brief(*, include_beta: bool) -> dict:
        papers = [{
            "id": "p1", **p1, "authors": [], "year": None,
            "evidence": [{"location": p1["id_or_url"],
                          "snippet": "alpha grounded evidence"}],
        }]
        claims = [{
            "id": "c1", "text": "Alpha claim", "evidence": [{
                "paper_id": "p1", "location": p1["id_or_url"],
                "snippet": "alpha grounded evidence",
            }],
        }]
        edges = [
            {"source": "d1", "target": "p1", "relation": "direction_has_paper"},
            {"source": "p1", "target": "c1", "relation": "paper_supports_claim"},
        ]
        key_papers = [p1]
        if include_beta:
            papers.append({
                "id": "p2", **p2, "authors": [], "year": None,
                "evidence": [{"location": p2["id_or_url"],
                              "snippet": "beta grounded evidence"}],
            })
            claims.append({
                "id": "c2", "text": "Beta claim", "evidence": [{
                    "paper_id": "p2", "location": p2["id_or_url"],
                    "snippet": "beta grounded evidence",
                }],
            })
            edges.extend([
                {"source": "d2", "target": "p2", "relation": "direction_has_paper"},
                {"source": "p2", "target": "c2", "relation": "paper_supports_claim"},
            ])
            key_papers.append(p2)
        return {
            "summary": "Grounded research directions.", "related_work": [],
            "key_papers": key_papers, "existing_benchmarks": [],
            "evidence_dag": {
                "directions": [
                    {"id": "d1", "label": "alpha", "basis": "user asks about alpha",
                     "source_refs": ["user:topic"]},
                    {"id": "d2", "label": "beta", "basis": "clarification asks about beta",
                     "source_refs": ["clarify:beta"]},
                ],
                "papers": papers, "claims": claims, "edges": edges,
            },
            "evidence_gaps": ([] if include_beta else [{
                "direction_id": "d2", "query": "beta mechanism",
                "reason": "no beta evidence", "attempted": False,
            }]),
        }

    backend = Backend()
    llm = FakeLLM([
        json.dumps(directions),
        _synth_json(brief(include_beta=False), VALID_FIELD_TRAJECTORY_BRIEF),
        _synth_json(brief(include_beta=True), VALID_FIELD_TRAJECTORY_BRIEF),
    ])

    result = deep_research_step(RESEARCH_SPEC, backend, llm)

    assert backend.beta_seminal_calls == 2
    assert result.literature_brief["evidence_gaps"] == []
    assert {item["id"] for item in result.literature_brief["evidence_dag"]["papers"]} == {
        "p1", "p2"}


def test_method_targeted_search_freezes_primary_core_before_retrieval():
    primary_core = {
        "core_id": "core-orthogonal-loss",
        "intervention": "orthogonalize the learned representation",
        "causal_mechanism": "separate factors before applying the task loss",
        "pipeline_role": "representation transformation",
        "expected_effect": "reduce interference in optimization",
    }
    plan = {
        "directions": [{
            "id": "d1", "label": "orthogonal representations",
            "query": "orthogonal representation task loss",
            "basis": "the extracted main method",
        }],
        "method_primary_core": primary_core,
    }
    class RecordingBackend(FakeBackend):
        def __init__(self):
            super().__init__(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
            self.seminal_queries = []

        def search_seminal(self, query, spec):
            self.seminal_queries.append(query)
            return super().search_seminal(query, spec)

    backend = RecordingBackend()
    llm = FakeLLM([
        json.dumps(plan),
        _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF),
    ])

    result = deep_research_step(
        RESEARCH_SPEC, backend, llm,
        target_method={"approach": "orthogonalize before the task loss"},
    )

    assert result.literature_brief["searched_method_primary_core"] == primary_core
    assert "orthogonal representation task loss" in backend.seminal_queries


# ---------------------------------------------------------------------------
# (c) THE key anti-hallucination test: LLM cites a paper never retrieved ->
#     core rejects/retries -> raises the typed parse error.
# ---------------------------------------------------------------------------


def test_ungrounded_key_paper_is_rejected_and_raises_typed_error():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    hallucinated_brief = {
        **VALID_LITERATURE_BRIEF,
        "key_papers": [
            SEMINAL_PAPERS[0],
            {
                "title": "A Paper That Was Never Retrieved",
                "id_or_url": "arxiv:9999.99999",
            },
        ],
    }
    llm = FakeLLM(
        [_PLAN_JSON]
        + [_synth_json(hallucinated_brief, VALID_FIELD_TRAJECTORY_BRIEF)] * (MAX_RETRIES + 1)
    )

    with pytest.raises(DeepResearchParseError):
        deep_research_step(RESEARCH_SPEC, backend, llm)

    # 1 keyword-planning call + (MAX_RETRIES + 1) synthesis attempts
    assert len(llm.calls) == MAX_RETRIES + 2


# ---------------------------------------------------------------------------
# (d) malformed LLM JSON -> bounded retry -> typed error.
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_raises_typed_error():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    llm = FakeLLM([_PLAN_JSON] + ["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(DeepResearchParseError):
        deep_research_step(RESEARCH_SPEC, backend, llm)

    # 1 keyword-planning call + (MAX_RETRIES + 1) synthesis attempts
    assert len(llm.calls) == MAX_RETRIES + 2


def test_synthesis_prompt_and_retry_history_are_bounded():
    long_excerpt = "paper-evidence-" * 2_000
    long_invalid = "not-json-" * 2_000
    paper = {**SEMINAL_PAPERS[0], "fetched_excerpt": long_excerpt}

    message = _build_user_message("query", RESEARCH_SPEC, [paper])
    assert long_excerpt not in message
    assert "paper-evidence-" in message

    backend = FakeBackend(seminal=[paper], frontier=[])
    llm = FakeLLM([_PLAN_JSON] + [long_invalid] * (MAX_RETRIES + 1))
    with pytest.raises(DeepResearchParseError):
        deep_research_step(RESEARCH_SPEC, backend, llm)

    retry_messages = llm.calls[2]
    assistant_echoes = [
        item["content"] for item in retry_messages if item["role"] == "assistant"
    ]
    assert len(assistant_echoes) == MAX_RETRIES + 1
    assert all(len(item) < len(long_invalid) for item in assistant_echoes)
    assert all("characters omitted from invalid response" in item
               for item in assistant_echoes)


def test_synthesis_requires_structured_output_from_the_client():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    llm = FakeLLM([
        _PLAN_JSON,
        _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF),
    ])

    deep_research_step(RESEARCH_SPEC, backend, llm)

    assert llm.response_formats == ["json_object", "json_object_required"]
    assert llm.max_tokens == [None, 262_144]


def test_invalid_json_reason_identifies_output_limit():
    reason = _invalid_json_reason(
        "{" * 100,
        {"completion_tokens": 262_144},
    )

    assert "chars=100" in reason
    assert "completion_tokens=262144" in reason
    assert "output limit" in reason


def test_evidence_gap_shorthand_is_normalized_before_schema_validation():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    brief = {
        **VALID_LITERATURE_BRIEF,
        "evidence_gaps": [{"query": "missing evidence", "note": "needs follow-up"}],
    }
    llm = FakeLLM([
        json.dumps({
            "directions": [{
                "id": "d1",
                "label": "scientific RAG",
                "query": "scientific RAG",
                "basis": "research topic",
            }]
        }),
        _synth_json(brief, VALID_FIELD_TRAJECTORY_BRIEF),
    ])

    result = deep_research_step(RESEARCH_SPEC, backend, llm)

    assert result.literature_brief["evidence_gaps"] == [{
        "direction_id": "d1",
        "query": "missing evidence",
        "reason": "needs follow-up",
        "attempted": True,
    }]


def test_repairable_synthesis_json_still_passes_full_contract_validation():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    malformed = _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF)
    malformed = malformed.replace('"related_work": [],', '"related_work": []', 1)
    llm = FakeLLM([_PLAN_JSON, malformed])

    result = deep_research_step(RESEARCH_SPEC, backend, llm)

    assert result.literature_brief["summary"] == VALID_LITERATURE_BRIEF["summary"]


def test_non_verbatim_evidence_quote_is_replaced_with_retrieved_source_text():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=[])
    brief = {
        **VALID_LITERATURE_BRIEF,
        "key_papers": [SEMINAL_PAPERS[0]],
        "evidence_dag": {
            "directions": [{
                "id": "d1", "label": "RAG", "basis": "research topic",
                "source_refs": ["research topic"],
            }],
            "papers": [{
                "id": "p1", "title": SEMINAL_PAPERS[0]["title"],
                "id_or_url": SEMINAL_PAPERS[0]["id_or_url"],
                "year": SEMINAL_PAPERS[0]["year"], "authors": [],
                "evidence": [{"location": "translated", "snippet": "这是翻译而非原文"}],
            }],
            "claims": [{
                "id": "c1", "text": "RAG grounds generation with retrieval",
                "evidence": [{
                    "paper_id": "p1", "location": "translated",
                    "snippet": "这是翻译而非原文",
                }],
            }],
            "edges": [
                {"source": "d1", "target": "p1", "relation": "direction_has_paper"},
                {"source": "p1", "target": "c1", "relation": "paper_supports_claim"},
            ],
        },
    }
    llm = FakeLLM([
        json.dumps({
            "directions": [{
                "id": "d1", "label": "RAG", "query": "RAG",
                "basis": "research topic",
            }]
        }),
        _synth_json(brief, VALID_FIELD_TRAJECTORY_BRIEF),
    ])

    result = deep_research_step(RESEARCH_SPEC, backend, llm)
    paper = result.literature_brief["evidence_dag"]["papers"][0]
    claim = result.literature_brief["evidence_dag"]["claims"][0]

    assert paper["evidence"][0]["snippet"] == SEMINAL_PAPERS[0]["note"]
    assert claim["evidence"][0]["snippet"] == paper["evidence"][0]["snippet"]


# ---------------------------------------------------------------------------
# (e) empty search results (both passes empty) -> valid sparse brief, no
#     crash, and (since there is nothing to ground) no SYNTHESIS LLM call is
#     made (the keyword-planning call still runs before retrieval).
# ---------------------------------------------------------------------------


def test_empty_search_results_returns_valid_sparse_brief_without_synthesis_call():
    backend = FakeBackend(seminal=[], frontier=[])
    llm = FakeLLM([_PLAN_JSON])  # keyword-planning only; synthesis must NOT run

    result = deep_research_step(RESEARCH_SPEC, backend, llm)

    assert isinstance(result, DeepResearchResult)
    assert result.literature_brief["key_papers"] == []

    lb_errors = list(LITERATURE_BRIEF_VALIDATOR.iter_errors(result.literature_brief))
    assert lb_errors == [], [e.message for e in lb_errors]

    ftb_errors = list(
        FIELD_TRAJECTORY_BRIEF_VALIDATOR.iter_errors(result.field_trajectory_brief)
    )
    assert ftb_errors == [], [e.message for e in ftb_errors]

    assert len(llm.calls) == 1  # only keyword-planning; no synthesis (nothing to ground)
    assert backend.seminal_calls == 1
    assert backend.frontier_calls == 1


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_no_prompt_patch_leaves_system_message_unchanged():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    llm = FakeLLM([_PLAN_JSON, _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF)])

    deep_research_step(RESEARCH_SPEC, backend, llm)

    messages = llm.calls[-1]  # the synthesis call (calls[0] is keyword-planning)
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_system_prompt()
    assert "translate, paraphrase, or normalize an evidence snippet" in system_message


def test_prompt_patch_appended_to_system_message():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    llm = FakeLLM([_PLAN_JSON, _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF)])

    deep_research_step(RESEARCH_SPEC, backend, llm, prompt_patch="STEER: prefer X")

    messages = llm.calls[-1]  # the synthesis call (calls[0] is keyword-planning)
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


# ---------------------------------------------------------------------------
# (f) real-LLM shape: response wrapped in a markdown ```json fence must still
# parse (proves core.py routes through extract_json, not a bare json.loads).
# ---------------------------------------------------------------------------


def test_deep_research_step_parses_response_wrapped_in_markdown_json_fence():
    backend = FakeBackend(seminal=SEMINAL_PAPERS, frontier=FRONTIER_PAPERS)
    fenced = (
        "```json\n"
        + _synth_json(VALID_LITERATURE_BRIEF, VALID_FIELD_TRAJECTORY_BRIEF)
        + "\n```"
    )
    llm = FakeLLM([_PLAN_JSON, fenced])

    result = deep_research_step(RESEARCH_SPEC, backend, llm)

    assert isinstance(result, DeepResearchResult)
    lb_errors = list(LITERATURE_BRIEF_VALIDATOR.iter_errors(result.literature_brief))
    assert lb_errors == [], [e.message for e in lb_errors]


# --- #8: EVERY topic (English included) is distilled to English keywords for OpenAlex,
#         and OpenAlex wildcard chars (?/*) are stripped so a raw question can't 400. ----

def test_keyword_query_english_topic_is_keyword_extracted():
    import json
    from autoresearch.agents.deep_research_agent.core import _keyword_query
    from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM
    # English topics are NO LONGER sent verbatim -- a whole question matches OpenAlex
    # poorly and its trailing '?' is read as a wildcard, so they too get distilled.
    llm = FakeLLM([json.dumps({"query": "bin packing first-fit approximation"})])
    q = _keyword_query({"topic": "Does First-Fit beat X on bin packing?"}, llm)
    assert q == "bin packing first-fit approximation"
    assert len(llm.calls) == 1


def test_keyword_query_chinese_topic_extracts_english_keywords():
    import json
    from autoresearch.agents.deep_research_agent.core import _keyword_query
    from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM
    llm = FakeLLM([json.dumps({"query": "0/1 knapsack greedy approximation ratio"})])
    q = _keyword_query({"topic": "在随机 0/1 背包问题上,价值密度贪心能否胜过顺序基线?"}, llm)
    assert q == "0/1 knapsack greedy approximation ratio"
    assert len(llm.calls) == 1


def test_keyword_query_fails_open_to_sanitized_raw_topic_on_bad_llm():
    from autoresearch.agents.deep_research_agent.core import _keyword_query
    from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM
    # unparseable LLM -> fall back to the raw topic, but with wildcard chars stripped
    llm = FakeLLM(["not json at all"])
    q = _keyword_query({"topic": "max independent set on random graphs?"}, llm)
    assert q == "max independent set on random graphs"   # trailing '?' stripped


def test_initial_search_plan_receives_goal_repository_and_constraints_not_only_topic():
    from autoresearch.agents.deep_research_agent.core import _research_directions
    from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM

    spec = {"topic": "optimize structured prediction", "research_goal": (
        "Use https://example.org/team/task-repository and its sequence task; "
        "the official evaluator defines the primary metric."),
        "budget": "one GPU", "forbidden_items": ["do not replace the selected task"]}
    llm = FakeLLM([json.dumps({"directions": [{
        "id": "protocol", "label": "official task semantics",
        "query": "task-repository sequence official evaluator metric direction",
        "basis": "the supplied task leaves the metric unresolved"}]})])

    directions = _research_directions(spec, llm)

    messages = llm.calls[0]
    assert json.dumps(spec, ensure_ascii=False) in messages[1]["content"]
    assert "authoritative documentation/evaluator" in messages[0]["content"]
    assert "a search query itself is not evidence" in messages[0]["content"]
    assert directions[0]["id"] == "protocol"


def test_sanitize_search_strips_openalex_wildcards():
    from autoresearch.agents.deep_research_agent.core import _sanitize_search
    assert _sanitize_search("k-NN accuracy?") == "k-NN accuracy"
    assert _sanitize_search("foo * bar ?") == "foo bar"
    assert "?" not in _sanitize_search("a? b* c?")
    assert "*" not in _sanitize_search("a? b* c?")
