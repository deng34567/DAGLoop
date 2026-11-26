"""
DeepResearchAgent core: the pure two-pass-search + brief-synthesis logic.
DeepResearchAgent 核心：与传输层无关的纯"两轮搜索 + 综述合成"逻辑。

Given a ResearchSpec, ``deep_research_step`` runs an injected
``DeepResearchBackend`` through two passes (seminal + frontier), calls an
injected LLM client to synthesize a ``LiteratureBrief`` + a
``FieldTrajectoryBrief`` grounded in the retrieved papers, and validates both
against their data-contract schemas — with NO EventBus / Redis / Store
access. The handler (a later milestone) is responsible for resolving refs,
persisting the briefs, and publishing events.

给定一个 ResearchSpec，``deep_research_step`` 通过注入的
``DeepResearchBackend`` 跑两轮搜索（seminal + frontier），调用注入的 LLM
客户端合成一份基于检索结果的 ``LiteratureBrief`` + ``FieldTrajectoryBrief``，
并对照 data-contract schema 校验两者 —— 不含任何 EventBus/Redis/Store 访问。
handler（后续里程碑）负责解析 ref、持久化两份 brief、发布事件。

Anti-hallucination grounding policy / 抗幻觉溯源策略
----------------------------------------------------
``key_papers`` (and ``related_work``) in the ``LiteratureBrief`` MUST cite
only papers that were actually returned by ``backend.search_seminal`` /
``backend.search_frontier``. This is enforced here, at parse time, as an
EXPLICIT check independent of the system prompt (the prompt asks nicely; this
module does not trust that it was obeyed): after parsing the LLM's JSON, every
``key_papers`` entry's ``id_or_url`` (falling back to ``title`` when no
``id_or_url`` is given) must appear in the retrieved (seminal + frontier) set.
An entry that fails this check is treated exactly like malformed JSON — the
LLM is given corrective feedback and retried (bounded), and after retries are
exhausted a ``DeepResearchParseError`` is raised.

``existing_benchmarks`` and the narrative fields of ``FieldTrajectoryBrief``
(``origin``, ``current_state``, ``main_method_families``,
``benchmark_evolution``, ``open_problems``) are DELIBERATELY NOT
grounding-checked against the retrieved paper set: benchmarks and field-level
historical context are broader domain knowledge the LLM may legitimately
synthesize from training knowledge rather than from the two search passes
(the two passes return *papers*, not benchmarks). Strict grounding is applied
to ``key_papers`` only, per the task's must-have.
``existing_benchmarks``、``FieldTrajectoryBrief`` 的叙述性字段不做溯源校验：
它们属于更广义的领域知识，两轮检索返回的是论文而非 benchmark 列表。严格溯源
仅施加于 ``key_papers``。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from jsonschema import Draft202012Validator
from json_repair import repair_json

from autoresearch.agents._reuse.json_extract import extract_json
from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class DeepResearchError(Exception):
    """Base for DeepResearchAgent core failures. / 核心失败基类。"""

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "deep_research_failed"
    #: transient | permanent
    classification: str = "transient"


class DeepResearchParseError(DeepResearchError):
    """LLM never returned a well-formed, schema-valid, grounded brief pair
    after retries. 重试用尽后 LLM 仍未返回合法、schema 校验通过且溯源正确的
    brief 对。"""

    code = "deep_research_parse_error"
    classification = "permanent"


class DeepResearchLLMError(DeepResearchError):
    """Infrastructure-level LLM failure (5xx / timeout). Transient, no
    internal retry. 基础设施层 LLM 失败（5xx/超时）。transient，不内部重试。"""

    code = "deep_research_llm_error"
    classification = "transient"


# ---------------------------------------------------------------------------
# LLM caller protocol / LLM 调用协议
# ---------------------------------------------------------------------------


class LLMResultLike(Protocol):
    content: str | None
    usage: dict
    cost_delta: str


class LLMClientLike(Protocol):
    """Subset of autoresearch.llm.HKGAIClient used here. / 这里用到的客户端子集。"""

    def chat(
        self,
        *,
        messages: list[dict],
        response_format: str | None = ...,
        max_tokens: int | None = ...,
    ) -> LLMResultLike: ...


class DeepResearchBackendLike(Protocol):
    """Subset of autoresearch.agents._reuse.deep_research_backend.DeepResearchBackend."""

    def search_seminal(self, query: str, spec: dict) -> list[dict]: ...

    def search_frontier(self, query: str, spec: dict) -> list[dict]: ...

    def fetch(self, papers: list[dict]) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Result object / 结果对象
# ---------------------------------------------------------------------------


@dataclass
class DeepResearchResult:
    """The result of one deep_research_step call. / 一次 deep_research_step 调用的结果。"""

    literature_brief: dict[str, Any]
    field_trajectory_brief: dict[str, Any]


# ---------------------------------------------------------------------------
# Static resources: system prompt + brief schema validators
# 静态资源：system prompt + brief schema 校验器
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "synthesize_system.md"
_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _literature_brief_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "literature_brief.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _field_trajectory_brief_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "field_trajectory_brief.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_literature_brief(brief: dict) -> list[str]:
    return [f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in _literature_brief_validator().iter_errors(brief)]


def _validate_field_trajectory_brief(brief: dict) -> list[str]:
    return [e.message for e in _field_trajectory_brief_validator().iter_errors(brief)]


# ---------------------------------------------------------------------------
# Query derivation / 检索 query 推导
# ---------------------------------------------------------------------------


def _derive_query(research_spec: dict) -> str:
    topic = research_spec.get("topic")
    if isinstance(topic, str) and topic.strip():
        return topic.strip()
    goal = research_spec.get("research_goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    return "unspecified research topic"


# A whole natural-language question matches an academic index POORLY -- and worse,
# OpenAlex reads '?' and '*' as WILDCARD operators, so a raw question ending in '?'
# 400s ("Wildcards (* or ?) require ..."). So we ALWAYS spend one LLM call to distil
# the topic (English OR any other language) into concise English keywords, then strip
# any wildcard chars. (Earlier this ran only for non-English topics, which let a long
# English question through verbatim and silently returned 0 papers.)
_KEYWORD_SYSTEM = (
    "You turn the user's research question plus its clarified ResearchSpec into a SMALL set of "
    "distinct literature directions. Directions are dynamic: do not force a fixed taxonomy. Each "
    "direction needs a concise English academic-search query and a short basis grounded in the "
    "question/spec. Reply with ONE JSON object: {\"directions\": [{\"id\": \"d1\", "
    "\"label\": \"<direction>\", \"query\": \"<3-8 English keywords>\", "
    "\"basis\": \"<why this direction follows from the question/clarification>\"}]}. "
    "When a target Method is supplied, first extract its main causal method and also return "
    "method_primary_core with exactly these non-empty fields: core_id, intervention, "
    "causal_mechanism, pipeline_role, expected_effect. Search directions must target that core, "
    "not its name, tuning, dataset, or secondary components. "
    "Legacy {\"query\": \"...\"} is also accepted as one direction."
    " For initial research of a user-specified benchmark or repository, use the full "
    "ResearchSpec, not just its short topic. When its deciding metric or direction is "
    "unresolved, include a query for that task's authoritative documentation/evaluator "
    "alongside the relevant method literature. Preserve supplied repository/task identities; "
    "do not substitute typical metrics from neighboring tasks. Retrieved primary-source "
    "evidence must determine the metric; a search query itself is not evidence."
)

# OpenAlex ?search= treats ? and * as wildcard operators -> a raw question 400s.
_SEARCH_WILDCARD_RE = re.compile(r"[?*]+")


def _sanitize_search(query: str) -> str:
    """Make a string safe + concise for an OpenAlex ``?search=``: strip the ?/*
    wildcard operators, collapse whitespace, and cap length."""
    s = _SEARCH_WILDCARD_RE.sub(" ", query or "")
    return re.sub(r"\s+", " ", s).strip()[:200]


def _fallback_primary_core(target_method: dict | None) -> dict[str, str] | None:
    if not isinstance(target_method, dict):
        return None
    components = [item for item in target_method.get("components") or []
                  if isinstance(item, dict)]
    first = components[0] if components else {}
    approach = str(target_method.get("approach") or first.get("mechanism") or "").strip()
    if not approach:
        return None
    mechanism = str(target_method.get("why_it_should_win")
                    or first.get("mechanism") or approach).strip()
    return {
        "core_id": "searched-primary-core",
        "intervention": approach,
        "causal_mechanism": mechanism,
        "pipeline_role": str(first.get("name") or "method pipeline").strip(),
        "expected_effect": str(first.get("expected_effect") or mechanism).strip(),
    }


def _research_plan(
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    target_method: dict | None = None,
    novelty_audit: dict | None = None,
    max_directions: int = 4,
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    """Extract the target main method, then derive bounded search directions."""
    base = _derive_query(research_spec)
    if target_method is not None:
        base = "\n".join([
            "Target method requiring exact prior-work search:",
            json.dumps(target_method, ensure_ascii=False)[:3000],
            "Research topic:",
            base,
            "Prior novelty audit gaps:",
            json.dumps(novelty_audit or {}, ensure_ascii=False)[:2000],
        ])
    try:
        result = llm_client.chat(
            messages=[{"role": "system", "content": _KEYWORD_SYSTEM},
                      {"role": "user", "content": (
                          f"Research topic:\n{base}\n\nClarified ResearchSpec (JSON):\n"
                          + json.dumps(research_spec, ensure_ascii=False))}],
            response_format="json_object")
        obj = json.loads(extract_json(result.content or ""))
        raw_directions = obj.get("directions")
        if not isinstance(raw_directions, list) and obj.get("query"):
            raw_directions = [{
                "id": "d1", "label": str(obj["query"]),
                "query": str(obj["query"]), "basis": "derived from the research topic",
            }]
        directions: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_directions or []):
            if not isinstance(item, dict):
                continue
            query = _sanitize_search(str(item.get("query") or ""))
            if not query or query.casefold() in seen:
                continue
            seen.add(query.casefold())
            raw_id = re.sub(r"[^a-z0-9]+", "-", str(item.get("id") or "").lower()).strip("-")
            directions.append({
                "id": raw_id or f"d{index + 1}",
                "label": str(item.get("label") or query).strip(),
                "query": query,
                "basis": str(item.get("basis") or "derived from the research topic").strip(),
            })
            if len(directions) >= max(1, int(max_directions)):
                break
        primary_core = obj.get("method_primary_core")
        if target_method is not None:
            required = ("core_id", "intervention", "causal_mechanism",
                        "pipeline_role", "expected_effect")
            if not isinstance(primary_core, dict) or not all(
                    isinstance(primary_core.get(key), str) and primary_core[key].strip()
                    for key in required):
                primary_core = _fallback_primary_core(target_method)
            else:
                primary_core = {key: primary_core[key].strip() for key in required}
        else:
            primary_core = None
        if directions:
            return directions, primary_core
    except Exception:  # noqa: BLE001 -- keyword extraction is best-effort; fall back
        pass
    query = _sanitize_search(base)
    return ([{"id": "d1", "label": query, "query": query,
              "basis": "fallback from the research topic"}],
            _fallback_primary_core(target_method))


def _research_directions(
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    target_method: dict | None = None,
    novelty_audit: dict | None = None,
    max_directions: int = 4,
) -> list[dict[str, str]]:
    """Backward-compatible direction-only helper."""
    return _research_plan(
        research_spec, llm_client, target_method=target_method,
        novelty_audit=novelty_audit, max_directions=max_directions)[0]


def _keyword_query(
    research_spec: dict,
    llm_client: LLMClientLike,
    *,
    target_method: dict | None = None,
    novelty_audit: dict | None = None,
) -> str:
    """Backward-compatible first direction query helper."""
    return _research_directions(
        research_spec, llm_client, target_method=target_method,
        novelty_audit=novelty_audit, max_directions=1)[0]["query"]


def _merge_direction_results(
    directions: list[dict[str, str]], backend: DeepResearchBackendLike,
    research_spec: dict, max_papers: int,
) -> list[dict]:
    by_key: dict[str, dict] = {}
    for direction in directions:
        papers = list(backend.search_seminal(direction["query"], research_spec))
        papers += list(backend.search_frontier(direction["query"], research_spec))
        for raw in papers:
            if not isinstance(raw, dict):
                continue
            key = _paper_key(raw)
            if not key:
                continue
            if key not in by_key:
                by_key[key] = dict(raw)
                by_key[key]["_direction_ids"] = []
            direction_ids = by_key[key]["_direction_ids"]
            if direction["id"] not in direction_ids:
                direction_ids.append(direction["id"])
            if len(by_key) >= max_papers:
                continue
    return list(by_key.values())[:max_papers]


def _fallback_evidence_dag(
    directions: list[dict[str, str]], retrieved: list[dict],
) -> dict[str, Any]:
    papers = []
    edges = []
    for index, paper in enumerate(retrieved):
        paper_id = f"p{index + 1}"
        snippet = str(
            paper.get("fetched_excerpt") or paper.get("note") or paper.get("title") or ""
        ).strip()[:1000]
        papers.append({
            "id": paper_id,
            "title": str(paper.get("title") or ""),
            "id_or_url": str(paper.get("id_or_url") or paper.get("title") or ""),
            "authors": [str(item) for item in (paper.get("authors") or []) if str(item)],
            "year": paper.get("year") if isinstance(paper.get("year"), int) else None,
            "evidence": [{
                "location": str(paper.get("id_or_url") or paper.get("title") or "retrieved"),
                "snippet": snippet,
            }],
        })
        for direction_id in paper.get("_direction_ids") or []:
            edges.append({"source": direction_id, "target": paper_id,
                          "relation": "direction_has_paper"})
    return {
        "directions": [
            {"id": item["id"], "label": item["label"], "basis": item["basis"],
             "source_refs": [item["basis"]]}
            for item in directions
        ],
        "papers": papers,
        "claims": [],
        "edges": edges,
    }


def _ground_dag_evidence_quotes(dag: Any, retrieved: list[dict]) -> None:
    """Replace non-verbatim model quotes with real text from the cited paper."""
    if not isinstance(dag, dict):
        return
    retrieved_by_key = {
        key: item for item in retrieved if (key := _paper_key(item))
    }
    paper_evidence_by_id: dict[str, list[dict[str, str]]] = {}
    for paper in dag.get("papers") or []:
        if not isinstance(paper, dict):
            continue
        source = retrieved_by_key.get(str(paper.get("id_or_url") or ""))
        if not source:
            continue
        source_parts = [
            str(source.get(key) or "").strip()
            for key in ("fetched_excerpt", "note", "title")
            if str(source.get(key) or "").strip()
        ]
        ground = "\n".join(source_parts)
        if not ground:
            continue
        fallback = source_parts[0][:_SYNTHESIS_EXCERPT_CHARS].strip()
        normalized: list[dict[str, str]] = []
        for evidence in paper.get("evidence") or []:
            item = evidence if isinstance(evidence, dict) else {}
            snippet = str(item.get("snippet") or "").strip()
            if not snippet or snippet not in ground:
                snippet = fallback
            normalized.append({
                "location": str(
                    item.get("location") or source.get("id_or_url")
                    or source.get("title") or "retrieved"
                ),
                "snippet": snippet,
            })
        if not normalized:
            normalized = [{
                "location": str(source.get("id_or_url") or source.get("title") or "retrieved"),
                "snippet": fallback,
            }]
        paper["evidence"] = normalized
        paper_evidence_by_id[str(paper.get("id") or "")] = normalized

    for claim in dag.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        normalized_claim_evidence: list[dict[str, str]] = []
        for evidence in claim.get("evidence") or []:
            item = evidence if isinstance(evidence, dict) else {}
            paper_id = str(item.get("paper_id") or "")
            paper_evidence = paper_evidence_by_id.get(paper_id) or []
            pair = (
                str(item.get("location") or ""),
                str(item.get("snippet") or ""),
            )
            exact = next((candidate for candidate in paper_evidence if (
                candidate["location"], candidate["snippet"]
            ) == pair), None)
            selected = exact or (paper_evidence[0] if paper_evidence else None)
            if selected is None:
                normalized_claim_evidence.append(dict(item))
                continue
            normalized_claim_evidence.append({
                "paper_id": paper_id,
                "location": selected["location"],
                "snippet": selected["snippet"],
            })
        claim["evidence"] = normalized_claim_evidence


def _validate_evidence_dag(
    dag: Any, directions: list[dict[str, str]], retrieved: list[dict],
) -> list[str]:
    if not isinstance(dag, dict):
        return ["evidence_dag must be an object"]
    direction_nodes = dag.get("directions") or []
    paper_nodes = dag.get("papers") or []
    claim_nodes = dag.get("claims") or []
    edges = dag.get("edges") or []
    if not all(isinstance(items, list) for items in (direction_nodes, paper_nodes, claim_nodes, edges)):
        return ["evidence_dag collections must be arrays"]
    groups = {"direction": direction_nodes, "paper": paper_nodes, "claim": claim_nodes}
    node_types: dict[str, str] = {}
    errors: list[str] = []
    for kind, nodes in groups.items():
        for node in nodes:
            node_id = str(node.get("id") or "") if isinstance(node, dict) else ""
            if not node_id or node_id in node_types:
                errors.append(f"duplicate or empty {kind} node ID: {node_id!r}")
            else:
                node_types[node_id] = kind
    expected_directions = {item["id"] for item in directions}
    if {str(item.get("id") or "") for item in direction_nodes if isinstance(item, dict)} \
            != expected_directions:
        errors.append("evidence_dag directions do not match the derived research directions")
    retrieved_by_key = {
        key: item for item in retrieved if (key := _paper_key(item))
    }
    retrieved_keys = set(retrieved_by_key)
    paper_ids: dict[str, dict] = {}
    for paper in paper_nodes:
        if not isinstance(paper, dict) or str(paper.get("id_or_url") or "") not in retrieved_keys:
            errors.append("evidence_dag paper is not grounded in retrieved results")
            continue
        paper_ids[str(paper.get("id") or "")] = paper
        retrieved_paper = retrieved_by_key[str(paper.get("id_or_url"))]
        ground = "\n".join(str(retrieved_paper.get(key) or "") for key in (
            "fetched_excerpt", "note", "title"))
        for evidence in paper.get("evidence") or []:
            snippet = str((evidence or {}).get("snippet") or "").strip()
            if snippet and snippet not in ground:
                errors.append(
                    f"evidence_dag paper snippet is not present in fetched evidence: {paper.get('id')}")
    relation_types = {
        "direction_has_paper": ("direction", "paper"),
        "paper_supports_claim": ("paper", "claim"),
        "claim_depends_on": ("claim", "claim"),
    }
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_types}
    supported_claims: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("evidence_dag edge must be an object")
            continue
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        relation = str(edge.get("relation") or "")
        expected = relation_types.get(relation)
        if source not in node_types or target not in node_types or expected is None:
            errors.append(f"invalid evidence_dag edge: {source}->{target} ({relation})")
            continue
        if (node_types[source], node_types[target]) != expected:
            errors.append(f"evidence_dag edge has wrong node types: {source}->{target} ({relation})")
            continue
        adjacency[source].append(target)
        if relation == "paper_supports_claim":
            supported_claims.add(target)
    missing_support = set(node_types) & {str(item.get("id") or "") for item in claim_nodes} \
        - supported_claims
    if missing_support:
        errors.append(f"claims lack retrieved-paper support: {sorted(missing_support)}")
    support_pairs = {
        (str(edge.get("source") or ""), str(edge.get("target") or ""))
        for edge in edges if isinstance(edge, dict)
        and edge.get("relation") == "paper_supports_claim"
    }
    normalized_claims: set[str] = set()
    for claim in claim_nodes:
        if not isinstance(claim, dict):
            continue
        normalized = re.sub(r"\W+", " ", str(claim.get("text") or "").casefold()).strip()
        if normalized in normalized_claims:
            errors.append("evidence_dag contains duplicate normalized claims")
        normalized_claims.add(normalized)
        claim_id = str(claim.get("id") or "")
        for evidence in claim.get("evidence") or []:
            paper_id = str((evidence or {}).get("paper_id") or "")
            if (paper_id, claim_id) not in support_pairs:
                errors.append(f"claim evidence lacks paper_supports_claim edge: {paper_id}->{claim_id}")
                continue
            paper = paper_ids.get(paper_id) or {}
            paper_evidence = {
                (str(item.get("location") or ""), str(item.get("snippet") or ""))
                for item in (paper.get("evidence") or [])
                if isinstance(item, dict)
            }
            claim_evidence = (
                str((evidence or {}).get("location") or ""),
                str((evidence or {}).get("snippet") or ""),
            )
            if claim_evidence not in paper_evidence:
                errors.append(f"claim evidence is not grounded in paper evidence: {paper_id}->{claim_id}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return False
        if node_id in visited:
            return True
        visiting.add(node_id)
        if any(not visit(child) for child in adjacency[node_id]):
            return False
        visiting.remove(node_id)
        visited.add(node_id)
        return True

    if any(not visit(node_id) for node_id in node_types):
        errors.append("evidence_dag contains a cycle")
    return errors


def _evidence_gaps(brief: dict, directions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return declared gaps plus any research direction with no grounded paper edge."""
    gaps = [dict(item) for item in (brief.get("evidence_gaps") or [])
            if isinstance(item, dict)]
    covered = {
        str(edge.get("source") or "")
        for edge in ((brief.get("evidence_dag") or {}).get("edges") or [])
        if isinstance(edge, dict) and edge.get("relation") == "direction_has_paper"
    }
    known = {str(item.get("direction_id") or "") for item in gaps}
    for direction in directions:
        if direction["id"] not in covered and direction["id"] not in known:
            gaps.append({
                "direction_id": direction["id"],
                "query": direction["query"],
                "reason": "no retrieved paper is attached to this research direction",
                "attempted": False,
            })
    return gaps


def _normalize_declared_evidence_gaps(
    gaps: Any,
    directions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Normalize model shorthand before schema validation; never invent a direction."""
    direction_by_id = {item["id"]: item for item in directions}
    normalized: list[dict[str, Any]] = []
    for raw in gaps if isinstance(gaps, list) else []:
        if not isinstance(raw, dict):
            continue
        direction_id = str(raw.get("direction_id") or "").strip()
        if not direction_id and len(directions) == 1:
            direction_id = directions[0]["id"]
        if not direction_id:
            continue
        direction = direction_by_id.get(direction_id) or {}
        query = str(raw.get("query") or direction.get("query") or "").strip()
        if not query:
            continue
        normalized.append({
            "direction_id": direction_id,
            "query": query,
            "reason": str(
                raw.get("reason") or raw.get("note") or "insufficient retrieved evidence"
            ).strip(),
            "attempted": bool(raw.get("attempted", False)),
        })
    return normalized


def _follow_up_search(
    gaps: list[dict[str, Any]], backend: DeepResearchBackendLike,
    research_spec: dict, retrieved: list[dict], max_papers: int,
) -> list[dict]:
    """Run one bounded targeted search and merge new papers into the grounded set."""
    by_key = {_paper_key(item): dict(item) for item in retrieved if _paper_key(item)}
    for gap in gaps:
        query = _sanitize_search(str(gap.get("query") or ""))
        direction_id = str(gap.get("direction_id") or "")
        if not query:
            continue
        papers = list(backend.search_seminal(query, research_spec))
        papers += list(backend.search_frontier(query, research_spec))
        for raw in papers:
            key = _paper_key(raw)
            if not key:
                continue
            paper = by_key.setdefault(key, dict(raw))
            paper.setdefault("_direction_ids", [])
            if direction_id and direction_id not in paper["_direction_ids"]:
                paper["_direction_ids"].append(direction_id)
            if len(by_key) >= max_papers:
                break
    merged = list(by_key.values())[:max_papers]
    fetch = getattr(backend, "fetch", None)
    if callable(fetch) and merged:
        fetched = fetch(merged)
        if isinstance(fetched, list):
            enriched = {_paper_key(item): item for item in fetched if _paper_key(item)}
            merged = [
                {**paper, **{
                    key: value for key, value in (enriched.get(_paper_key(paper)) or {}).items()
                    if key not in {"title", "id_or_url", "_direction_ids"}
                }} for paper in merged
            ]
    return merged


# ---------------------------------------------------------------------------
# Grounding check / 溯源校验
# ---------------------------------------------------------------------------


def _paper_key(paper: Any) -> str | None:
    """Return the identity key for a Paper dict: id_or_url, falling back to
    title. None if neither is a usable string.
    返回 Paper dict 的身份 key：id_or_url，缺失则回退 title；两者都没有则
    None。"""
    if not isinstance(paper, dict):
        return None
    key = paper.get("id_or_url") or paper.get("title")
    return key if isinstance(key, str) and key else None


def _find_ungrounded_key_papers(key_papers: list, retrieved: list[dict]) -> list[str]:
    """Return the keys of any key_papers entries not present in the retrieved
    (seminal + frontier) set. Empty list = fully grounded.
    返回未出现在检索集合（seminal + frontier）中的 key_papers 条目的 key
    列表。空列表 = 完全溯源。"""
    retrieved_keys = {k for p in retrieved if (k := _paper_key(p)) is not None}
    ungrounded: list[str] = []
    for paper in key_papers:
        key = _paper_key(paper)
        if key is None or key not in retrieved_keys:
            ungrounded.append(key if key is not None else repr(paper))
    return ungrounded


# ---------------------------------------------------------------------------
# Message building / 消息组装
# ---------------------------------------------------------------------------

_SYNTHESIS_EXCERPT_CHARS = 4_000
_SYNTHESIS_MAX_TOKENS = 262_144
_RETRY_RESPONSE_CHARS = 2_000


def _bounded_papers_for_prompt(retrieved: list[dict]) -> list[dict]:
    """Keep synthesis context bounded without changing persisted evidence."""
    bounded: list[dict] = []
    for paper in retrieved:
        item = dict(paper)
        excerpt = item.get("fetched_excerpt")
        if isinstance(excerpt, str) and len(excerpt) > _SYNTHESIS_EXCERPT_CHARS:
            item["fetched_excerpt"] = excerpt[:_SYNTHESIS_EXCERPT_CHARS]
        bounded.append(item)
    return bounded


def _bounded_retry_response(content: str) -> str:
    """Keep correction evidence while preventing retry-context amplification."""
    if len(content) <= _RETRY_RESPONSE_CHARS:
        return content
    half = _RETRY_RESPONSE_CHARS // 2
    omitted = len(content) - (half * 2)
    return (
        content[:half]
        + f"\n... [{omitted} characters omitted from invalid response] ...\n"
        + content[-half:]
    )


def _append_retry_feedback(messages: list[dict], content: str, reason: str) -> None:
    messages.append({"role": "assistant", "content": _bounded_retry_response(content)})
    messages.append(_retry_feedback_message(reason))


def _invalid_json_reason(content: str, usage: Any) -> str:
    completion_tokens = 0
    if isinstance(usage, dict):
        try:
            completion_tokens = int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            completion_tokens = 0
    reason = (
        "response was not valid JSON "
        f"(chars={len(content)}, completion_tokens={completion_tokens})"
    )
    if completion_tokens >= int(_SYNTHESIS_MAX_TOKENS * 0.95):
        reason += (
            "; response likely hit the output limit: select only the strongest "
            "non-redundant evidence and make the complete JSON substantially shorter"
        )
    return reason


def _parse_synthesis_json(content: str) -> Any:
    raw = extract_json(content)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return repair_json(raw, return_objects=True)


def _build_user_message(
    query: str,
    research_spec: dict,
    retrieved: list[dict],
    *,
    directions: list[dict[str, str]] | None = None,
    target_method: dict | None = None,
    novelty_audit: dict | None = None,
) -> str:
    lines = [
            f"Research query: {query}",
            "",
            "ResearchSpec (JSON):",
            json.dumps(research_spec, ensure_ascii=False, indent=2),
            "",
            "Research directions derived from the user question and clarification (JSON):",
            json.dumps(directions or [], ensure_ascii=False, indent=2),
    ]
    if target_method is not None:
        lines += [
            "",
            "Frozen method whose novelty is being re-audited (JSON):",
            json.dumps(target_method, ensure_ascii=False, indent=2),
            "",
            "Prior novelty audit gaps to resolve (JSON):",
            json.dumps(novelty_audit or {}, ensure_ascii=False, indent=2),
        ]
    lines += [
            "",
            f"Retrieved papers (seminal + frontier, {len(retrieved)} total, JSON):",
            json.dumps(_bounded_papers_for_prompt(retrieved), ensure_ascii=False, indent=2),
            "",
            "Synthesize a LiteratureBrief and FieldTrajectoryBrief using ONLY "
            "the papers listed above for `key_papers` / `related_work`. "
            "Omit optional year in key_papers / related_work when unknown; "
            "evidence_dag.papers requires year and permits null for an unknown year. "
            "Build evidence_dag as direction -> paper -> claim. Preserve shared papers across "
            "directions and merge the same claim when several retrieved papers support it. "
            'Respond with a single JSON object: {"literature_brief": {...}, '
            '"field_trajectory_brief": {...}}.',
        ]
    return "\n".join(lines)


def _retry_feedback_message(reason: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}\n"
            "Respond again with ONLY a single JSON object matching the "
            'required {"literature_brief": {...}, "field_trajectory_brief": '
            '{...}} shape described in the system prompt. Do not invent '
            "papers that were not in the retrieved list."
        ),
    }


# ---------------------------------------------------------------------------
# Sparse fallback for empty search results / 空检索结果的稀疏兜底
# ---------------------------------------------------------------------------


def _sparse_result(
    query: str,
    directions: list[dict[str, str]],
    method_primary_core: dict[str, str] | None = None,
) -> DeepResearchResult:
    """Deterministic, LLM-free result for when both search passes returned no
    papers: a valid sparse brief signaling insufficient evidence, never a
    crash and never a fabricated citation.
    两轮检索都无结果时的确定性、无需 LLM 的兜底：一份合法的稀疏 brief，
    标示证据不足，绝不崩溃、也绝不虚构引用。"""
    literature_brief = {
        "summary": (
            f"No papers were retrieved for query {query!r} across either the "
            "seminal or frontier search pass (insufficient evidence)."
        ),
        "related_work": [],
        "key_papers": [],
        "existing_benchmarks": [],
        "evidence_dag": _fallback_evidence_dag(directions, []),
        "evidence_gaps": [
            {"direction_id": item["id"], "query": item["query"],
             "reason": "no paper was retrieved for this direction", "attempted": True}
            for item in directions
        ],
    }
    if method_primary_core is not None:
        literature_brief["searched_method_primary_core"] = method_primary_core
    field_trajectory_brief = {
        "origin": "Unknown: insufficient evidence (no papers retrieved).",
        "current_state": "Unknown: insufficient evidence (no papers retrieved).",
        "main_method_families": [],
        "open_problems": ["Insufficient evidence: literature search returned no papers."],
    }
    return DeepResearchResult(
        literature_brief=literature_brief, field_trajectory_brief=field_trajectory_brief
    )


# ---------------------------------------------------------------------------
# Pipeline / 主流程
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # malformed synthesis retries
MAX_GAP_ROUNDS = 1  # one bounded targeted follow-up search before freezing honest gaps


def deep_research_step(
    research_spec: dict,
    backend: DeepResearchBackendLike,
    llm_client: LLMClientLike,
    max_papers: int = 20,
    *,
    prompt_patch: str | None = None,
    target_method: dict | None = None,
    novelty_audit: dict | None = None,
    max_directions: int = 4,
) -> DeepResearchResult:
    """Run one two-pass search + synthesis step (pure, no transport).
    跑一次两轮搜索 + 合成（纯，无传输层）。

    Args:
        research_spec: the (dict-shaped) ResearchSpec to derive a query from.
        backend: object exposing ``search_seminal``/``search_frontier``.
        llm_client: object exposing ``.chat(messages=..., response_format=...)``.
        max_papers: cap on the number of retrieved papers (seminal + frontier
            combined) fed to the LLM / used for the grounding check.
        prompt_patch: optional MOSS-injected guidance appended to the system
            prompt (see ``autoresearch.agents._reuse.prompt_patch``).

    Returns:
        DeepResearchResult with schema-valid ``literature_brief`` and
        ``field_trajectory_brief``, ``key_papers`` grounded in the retrieved
        set.

    Raises:
        DeepResearchParseError (permanent): malformed/unparseable LLM JSON,
            a schema-invalid brief, or an ungrounded ``key_papers`` entry,
            after bounded retries.
        DeepResearchLLMError (transient): the LLM client itself raised.
    """
    directions, method_primary_core = _research_plan(
        research_spec, llm_client,
        target_method=target_method, novelty_audit=novelty_audit,
        max_directions=max_directions)
    query = "; ".join(item["query"] for item in directions)
    retrieved = _merge_direction_results(directions, backend, research_spec, max_papers)
    fetch = getattr(backend, "fetch", None)
    if callable(fetch) and retrieved:
        try:
            fetched = fetch(retrieved)
            if isinstance(fetched, list):
                by_key = {_paper_key(item): item for item in fetched if _paper_key(item)}
                retrieved = [
                    {**paper, **{
                        key: value for key, value in (by_key.get(_paper_key(paper)) or {}).items()
                        if key not in {"title", "id_or_url", "_direction_ids"}
                    }}
                    for paper in retrieved
                ]
        except Exception:  # noqa: BLE001 -- fetch enriches evidence; search remains usable
            pass

    if not retrieved:
        # No evidence at all: never call the LLM (nothing to ground against;
        # any brief it produced would necessarily be fabricated). / 无证据：
        # 绝不调用 LLM（无可溯源对象，其产出必然是虚构的）。
        return _sparse_result(query, directions, method_primary_core)

    messages = [
        {"role": "system", "content": apply_prompt_patch(_load_system_prompt(), prompt_patch)},
        {"role": "user", "content": _build_user_message(
            query, research_spec, retrieved,
            directions=directions,
            target_method=target_method, novelty_audit=novelty_audit)},
    ]

    last_error = "no attempts made"

    gap_round = 0
    synthesis_attempt = 0
    synthesis_budget = MAX_RETRIES + 1
    while synthesis_attempt < synthesis_budget:
        synthesis_attempt += 1
        try:
            result = llm_client.chat(
                messages=messages,
                response_format="json_object_required",
                max_tokens=_SYNTHESIS_MAX_TOKENS,
            )
        except Exception as exc:  # any client exception is infra/transient
            raise DeepResearchLLMError(
                f"LLM call failed: {type(exc).__name__}: {exc}"
            ) from exc

        content = result.content or ""

        try:
            obj = _parse_synthesis_json(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            last_error = _invalid_json_reason(content, result.usage)
            _append_retry_feedback(messages, content, last_error)
            continue

        if (
            not isinstance(obj, dict)
            or "literature_brief" not in obj
            or "field_trajectory_brief" not in obj
        ):
            last_error = (
                "response JSON is missing the required 'literature_brief' "
                "or 'field_trajectory_brief' key"
            )
            _append_retry_feedback(messages, content, last_error)
            continue

        literature_brief = obj.get("literature_brief")
        field_trajectory_brief = obj.get("field_trajectory_brief")

        if not isinstance(literature_brief, dict) or not isinstance(
            field_trajectory_brief, dict
        ):
            last_error = (
                "'literature_brief' and 'field_trajectory_brief' must both "
                "be JSON objects"
            )
            _append_retry_feedback(messages, content, last_error)
            continue

        literature_brief = dict(literature_brief)
        if method_primary_core is not None:
            literature_brief["searched_method_primary_core"] = method_primary_core
        literature_brief.setdefault(
            "evidence_dag", _fallback_evidence_dag(directions, retrieved))
        _ground_dag_evidence_quotes(literature_brief.get("evidence_dag"), retrieved)
        literature_brief["evidence_gaps"] = _normalize_declared_evidence_gaps(
            literature_brief.get("evidence_gaps"),
            directions,
        )

        lb_errors = _validate_literature_brief(literature_brief)
        if lb_errors:
            last_error = f"literature_brief failed schema validation: {lb_errors}"
            _append_retry_feedback(messages, content, last_error)
            continue

        gaps = _evidence_gaps(literature_brief, directions)
        unknown_gap_directions = sorted({
            str(item.get("direction_id") or "") for item in gaps
        } - {item["id"] for item in directions})
        if unknown_gap_directions:
            last_error = (
                "evidence_gaps reference unknown research directions: "
                f"{unknown_gap_directions}")
            _append_retry_feedback(messages, content, last_error)
            continue
        literature_brief["evidence_gaps"] = gaps
        if gaps and gap_round < MAX_GAP_ROUNDS:
            try:
                supplemented = _follow_up_search(
                    gaps, backend, research_spec, retrieved, max_papers)
            except Exception:  # noqa: BLE001 -- preserve an honest attempted gap
                supplemented = retrieved
            gap_round += 1
            if supplemented != retrieved:
                retrieved = supplemented
                # A successful evidence-gap search earns exactly one fresh synthesis.  It must
                # not silently enlarge the retry budget for malformed or ungrounded responses.
                synthesis_budget += 1
                messages.append({
                    "role": "assistant",
                    "content": _bounded_retry_response(content),
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "A bounded follow-up search was run for every evidence_gaps item. "
                        "Rebuild the complete two briefs from this supplemented retrieved-paper "
                        "set. Mark any remaining gap attempted=true; never invent evidence:\n"
                        + json.dumps(
                            _bounded_papers_for_prompt(retrieved),
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                })
                continue
        if gaps:
            literature_brief["evidence_gaps"] = [
                {**item, "attempted": True} for item in gaps
            ]

        ftb_errors = _validate_field_trajectory_brief(field_trajectory_brief)
        if ftb_errors:
            last_error = (
                f"field_trajectory_brief failed schema validation: {ftb_errors}"
            )
            _append_retry_feedback(messages, content, last_error)
            continue

        # EXPLICIT anti-hallucination grounding check — the enforceable
        # guard; do NOT trust that the prompt's instruction was followed.
        ungrounded = _find_ungrounded_key_papers(
            literature_brief.get("key_papers", []), retrieved
        )
        if ungrounded:
            last_error = (
                "key_papers cited papers not present in the retrieved "
                f"(seminal + frontier) set: {ungrounded}. Only cite papers "
                "from the retrieved list; do not invent papers."
            )
            _append_retry_feedback(messages, content, last_error)
            continue

        dag_errors = _validate_evidence_dag(
            literature_brief.get("evidence_dag"), directions, retrieved)
        if dag_errors:
            last_error = f"evidence_dag failed grounding/graph validation: {dag_errors}"
            _append_retry_feedback(messages, content, last_error)
            continue

        return DeepResearchResult(
            literature_brief=literature_brief,
            field_trajectory_brief=field_trajectory_brief,
        )

    raise DeepResearchParseError(
        "LLM did not produce a valid, grounded literature brief after "
        f"retries: {last_error}"
    )


__all__ = [
    "DeepResearchError",
    "DeepResearchParseError",
    "DeepResearchLLMError",
    "DeepResearchResult",
    "deep_research_step",
    "MAX_RETRIES",
    "MAX_GAP_ROUNDS",
]
