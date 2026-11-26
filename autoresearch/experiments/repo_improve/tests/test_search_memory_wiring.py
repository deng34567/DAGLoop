"""组1 接线: runner 每轮写经验卡 + method_agent 换方法读回搜索足迹。"""
from __future__ import annotations

from autoresearch.agents.method_agent.handler import MethodAgentHandler
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.experiments.repo_improve.search_memory import (
    ExperienceCard, board_for_session)


def _runner() -> RunnerAgentHandler:
    return RunnerAgentHandler(spec_store=None, backend=None)


def test_runner_records_card_on_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    _runner()._record_experience_card("sess1234", {
        "verdict": "invalid", "reasons": ["RuntimeError: CUDA out of memory"],
        "method_metric": None, "baseline_metric": None})
    cards = board_for_session("sess1234").cards()
    assert len(cards) == 1
    assert cards[0].valid is False and cards[0].verdict == "invalid"
    assert "RuntimeError" in cards[0].error_signature


def test_runner_records_valid_card_no_signature_and_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    r = _runner()
    r._record_experience_card("s", {"verdict": "invalid", "reasons": ["ValueError: bad"],
                                    "method_metric": None, "baseline_metric": 1.7})
    r._record_experience_card("s", {"verdict": "new_best", "reasons": [],
                                    "method_metric": 1.59, "baseline_metric": 1.7})
    cards = board_for_session("s").cards()
    assert len(cards) == 2                               # appends, not overwrites
    assert cards[1].valid is True and cards[1].metric_value == 1.59
    assert cards[1].error_signature == ""               # valid -> no signature
    assert cards[0].metric_value is None               # a missing measurement stays missing


def test_runner_records_partial_implementation_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    _runner()._record_experience_card("s", {
        "verdict": "valid_not_best",
        "reasons": [],
        "method_metric": 1.2,
        "baseline_metric": 1.0,
        "method_implementation_review": {"verdict": "partial"},
    })

    card = board_for_session("s").cards()[0]

    assert card.valid is True
    assert card.implementation_verdict == "partial"


def test_component_card_does_not_label_non_best_attempt_new_best(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    _runner()._record_experience_card("s", {
        "verdict": "new_best",
        "reasons": [],
        "method_metric": 2.29,
        "baseline_metric": 2.55,
        "component_execution": {
            "mode": "component",
            "component_id": "scalar-unroll",
            "became_best": False,
        },
    })

    card = board_for_session("s").cards()[0]

    assert card.valid is True
    assert card.verdict == "valid_not_best"


def test_record_card_never_raises_on_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    _runner()._record_experience_card("s", {})           # empty gen_result -> no crash
    _runner()._record_experience_card("s", {"reasons": None})


def test_method_agent_reads_search_history(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    board = board_for_session("sessX")
    board.add_card(ExperienceCard(round=1, method_family="lr-schedule", metric_value=1.7,
                                  valid=True, lower_is_better=True))
    board.add_card(ExperienceCard(round=2, valid=False,
                                  error_signature="RuntimeError: oom"))
    h = MethodAgentHandler(spec_store=None, llm_client=None)
    hist = h._search_history("sessX")
    assert "lr-schedule" in hist and "失败" in hist


def test_method_agent_search_history_empty_when_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    h = MethodAgentHandler(spec_store=None, llm_client=None)
    assert h._search_history("nobody") == ""


def test_rejected_method_definition_survives_later_regeneration(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    board = board_for_session("sess-rejected")
    rejected = {"approach": "replace optimizer mechanism", "changes_outline": ["change x"]}

    board.remember_rejected_method(rejected)
    board.remember_rejected_method(rejected)

    history = MethodAgentHandler(spec_store=None, llm_client=None)._search_history(
        "sess-rejected")
    assert history.count("replace optimizer mechanism") == 1


# --- 组2: _with_operator prefixes the request from the search trace ---------------------

def test_with_operator_prepends_debug_preamble(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "1")
    board_for_session("sopA").add_card(ExperienceCard(
        round=1, valid=False, error_signature="RuntimeError: boom"))
    out = _runner()._with_operator("sopA", "improve nanoGPT")
    assert out.startswith("[本轮算子=DEBUG]") and out.endswith("improve nanoGPT")


def test_with_operator_draft_returns_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "1")
    assert _runner()._with_operator("empty", "GOAL") == "GOAL"     # empty board -> draft


def test_with_operator_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_OPERATORS", "0")
    board_for_session("sopB").add_card(ExperienceCard(
        round=1, valid=False, error_signature="RuntimeError: boom"))
    assert _runner()._with_operator("sopB", "GOAL") == "GOAL"      # flag off -> unchanged
