"""Search experience memory (借鉴 Frontis-MA1): cards + board + stable error signatures."""
from __future__ import annotations

import os

from autoresearch.experiments.repo_improve.search_memory import (
    ExperienceCard, SearchBoard, board_for_workroot, decide_operator, normalize_error,
    prior_worktree_for_next_operator)


# --- normalize_error: same class of error -> same signature ---------------------------

def test_normalize_strips_paths_lineno_numbers_hex_uuid():
    a = normalize_error("FileNotFoundError: [Errno 2] No such file: '/tmp/abc-123/repo/train.py'")
    b = normalize_error("FileNotFoundError: [Errno 2] No such file: '/data/xyz/other/repo/train.py'")
    assert a == b                                   # different paths -> same signature
    assert "FileNotFoundError" in a and "<path>" in a


def test_normalize_same_signature_across_addresses_and_lines():
    a = normalize_error("RuntimeError: boom at 0xdeadbeef line 42")
    b = normalize_error("RuntimeError: boom at 0xcafef00d line 999")
    assert a == b
    assert a.startswith("RuntimeError")


def test_normalize_empty_is_empty():
    assert normalize_error("") == ""
    assert normalize_error(None) == ""


def test_normalize_distinguishes_different_errors():
    assert normalize_error("ModuleNotFoundError: No module named 'torch'") != \
           normalize_error("RuntimeError: CUDA out of memory")


# --- SearchBoard: persistence + aggregation -------------------------------------------

def _board(tmp_path) -> SearchBoard:
    return SearchBoard(os.path.join(str(tmp_path), "sm.json"))


def test_add_and_reload_cards_persist(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, operator="draft", method_family="lr", metric_value=1.5,
                              valid=True, verdict="valid_not_best"))
    # a fresh board over the SAME file sees the card (round-trip through disk)
    assert SearchBoard(b.path).cards()[0].method_family == "lr"


def test_missing_or_corrupt_file_degrades_to_empty(tmp_path):
    assert _board(tmp_path).cards() == []            # missing
    p = os.path.join(str(tmp_path), "bad.json")
    with open(p, "w") as fh:
        fh.write("{not json")
    assert SearchBoard(p).cards() == []              # corrupt -> empty, no raise


def test_new_method_archives_prior_cards_and_starts_with_draft(tmp_path):
    b = _board(tmp_path)
    b.begin_method("sha256:old")
    b.add_card(ExperienceCard(round=1, operator="improve", method_family="curriculum",
                              metric_value=1.2, valid=True, verdict="valid_not_best"))

    b.begin_method("sha256:revised")

    assert b.cards() == []
    assert decide_operator(b) == ("draft", "")
    assert "curriculum(best=1.2)" in b.summarize_for_regen()


def test_replaying_same_method_does_not_clear_current_cards(tmp_path):
    b = _board(tmp_path)
    b.begin_method("sha256:same")
    b.add_card(ExperienceCard(round=1, operator="draft", method_family="lr-finder",
                              metric_value=1.1, valid=True, verdict="new_best"))

    b.begin_method("sha256:same")

    assert [card.method_family for card in b.cards()] == ["lr-finder"]


def test_method_families_tried_dedup_ordered(tmp_path):
    b = _board(tmp_path)
    for fam in ["lr", "tokenizer", "lr", "", "depth"]:
        b.add_card(ExperienceCard(round=1, method_family=fam))
    assert b.method_families_tried() == ["lr", "tokenizer", "depth"]


def test_best_per_family_respects_direction(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, method_family="lr", metric_value=1.6, valid=True,
                              lower_is_better=True))
    b.add_card(ExperienceCard(round=2, method_family="lr", metric_value=1.4, valid=True,
                              lower_is_better=True))
    b.add_card(ExperienceCard(round=3, method_family="lr", metric_value=9.9, valid=False,
                              lower_is_better=True))   # invalid -> ignored
    assert b.best_per_family()["lr"].metric_value == 1.4   # lower is better


def test_best_per_family_higher_is_better(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, method_family="acc", metric_value=0.90, valid=True,
                              lower_is_better=False))
    b.add_card(ExperienceCard(round=2, method_family="acc", metric_value=0.95, valid=True,
                              lower_is_better=False))
    assert b.best_per_family()["acc"].metric_value == 0.95


def test_failure_patterns_counts_and_by_signature(tmp_path):
    b = _board(tmp_path)
    sig = normalize_error("RuntimeError: CUDA out of memory (tried 2GB)")
    for _ in range(3):
        b.add_card(ExperienceCard(round=1, error_signature=sig, valid=False))
    b.add_card(ExperienceCard(round=2, error_signature=normalize_error("ValueError: bad shape"),
                              valid=False))
    patterns = b.failure_patterns()
    assert patterns[0][0] == sig and patterns[0][1] == 3      # most frequent first
    assert len(b.by_error_signature(sig)) == 3
    assert b.by_error_signature("") == []


def test_summarize_for_regen_mentions_families_and_failures(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, method_family="lr", metric_value=1.4, valid=True,
                              lower_is_better=True))
    b.add_card(ExperienceCard(round=2, error_signature=normalize_error("RuntimeError: boom"),
                              valid=False))
    s = b.summarize_for_regen()
    assert "lr" in s and "best=1.4" in s and "失败" in s


def test_summarize_empty_when_nothing(tmp_path):
    assert _board(tmp_path).summarize_for_regen() == ""


def test_board_for_workroot_places_file_in_workroot(tmp_path):
    b = board_for_workroot(str(tmp_path))
    assert b.path == os.path.join(str(tmp_path), "search_memory.json")


# --- decide_operator: Draft/Improve/Debug/Crossover from the search trace ---------------

def test_decide_operator_empty_is_draft(tmp_path):
    op, pre = decide_operator(_board(tmp_path))
    assert op == "draft" and pre == ""


def test_decide_operator_last_failure_is_debug(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=1.7))
    b.add_card(ExperienceCard(round=2, valid=False,
                              error_signature="RuntimeError: cuda out of memory"))
    op, pre = decide_operator(b)
    assert op == "debug"
    assert "DEBUG" in pre and "cuda out of memory" in pre


def test_decide_operator_ignores_conversation_infra_failure(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(
        round=1, valid=True, metric_value=1.7, verdict="valid_not_best"))
    b.add_card(ExperienceCard(
        round=2,
        valid=False,
        error_signature=(
            "APIError: conversation run failed for id=<id>: output text may contain "
            "sensitive information"),
    ))

    op, pre = decide_operator(b)

    assert op == "check-expectation"
    assert "conversation run failed" not in pre


def test_decide_operator_legacy_crossover_env_still_improves(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_ENABLE_CROSSOVER", "1")
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=1.6, lower_is_better=True))
    b.add_card(ExperienceCard(round=2, valid=True, metric_value=1.5, lower_is_better=True))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "1.5" in pre and "CROSSOVER" not in pre


def test_decide_operator_one_valid_is_improve(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=1.6, lower_is_better=True))
    op, pre = decide_operator(b)
    assert op == "improve" and "1.6" in pre


def test_decide_operator_improve_orders_best_first_for_direction(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_ENABLE_CROSSOVER", "1")
    # higher-is-better: best (largest) should be quoted first
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=0.88, lower_is_better=False))
    b.add_card(ExperienceCard(round=2, valid=True, metric_value=0.95, lower_is_better=False))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "0.95" in pre and "0.88" not in pre


# --- P1-2: 单次决策只读盘一次(懒/省开销) ------------------------------------------------

def _load_counting_board(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, method_family="lr", metric_value=1.6, valid=True))
    b.add_card(ExperienceCard(round=2, valid=False, error_signature="RuntimeError: boom"))
    calls = {"n": 0}
    real_load = b._load

    def counting_load():
        calls["n"] += 1
        return real_load()
    b._load = counting_load
    return b, calls


def test_decide_operator_reads_disk_once(tmp_path):
    b, calls = _load_counting_board(tmp_path)
    decide_operator(b)
    assert calls["n"] == 1             # one load per decision, not re-read per aggregation


def test_summarize_for_regen_reads_disk_once(tmp_path):
    b, calls = _load_counting_board(tmp_path)
    b.summarize_for_regen()
    assert calls["n"] == 1


# --- P0: 算子把上一轮/前几轮的真实 diff 带给下一轮(累积 + 真Crossover) ---------------------

def _board(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import SearchBoard
    return SearchBoard(str(tmp_path / "sm.json"))


def test_card_roundtrips_diff():
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard
    c = ExperienceCard(round=1, diff="--- a/x\n+++ b/x\n+lr=3e-4")
    assert ExperienceCard.from_dict(c.to_dict()).diff == "--- a/x\n+++ b/x\n+lr=3e-4"


def test_improve_carries_previous_best_diff(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, verdict="new_best", metric_value=1.45,
                              lower_is_better=True, diff="+++ b/train.py\n+lr = 3e-4"))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "先把它应用到代码里" in pre       # 让 agent 先复现上一轮
    assert "lr = 3e-4" in pre               # 真实 diff 进了提示


def test_legacy_crossover_env_keeps_only_best_real_diff(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    monkeypatch.setenv("AUTORESEARCH_ENABLE_CROSSOVER", "1")
    monkeypatch.setenv("AUTORESEARCH_CROSSOVER_VARIANTS", "3")
    b = _board(tmp_path)
    for r, (v, d) in enumerate([(1.45, "DIFF_A"), (1.44, "DIFF_B"), (1.46, "DIFF_C")], start=1):
        b.add_card(ExperienceCard(round=r, valid=True, verdict="new_best", metric_value=v,
                                  lower_is_better=True, diff=d))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "DIFF_B" in pre and "DIFF_A" not in pre and "DIFF_C" not in pre
    assert "实际 diff" in pre


def test_legacy_crossover_variant_env_does_not_activate_crossover(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    monkeypatch.setenv("AUTORESEARCH_ENABLE_CROSSOVER", "1")
    monkeypatch.setenv("AUTORESEARCH_CROSSOVER_VARIANTS", "2")   # 只拼最优2个
    b = _board(tmp_path)
    for r, (v, d) in enumerate([(1.45, "DIFF_A"), (1.44, "DIFF_B"), (1.46, "DIFF_C")], start=1):
        b.add_card(ExperienceCard(round=r, valid=True, verdict="new_best", metric_value=v,
                                  lower_is_better=True, diff=d))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "DIFF_B" in pre and "DIFF_A" not in pre and "DIFF_C" not in pre


def test_debug_carries_failed_round_diff(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=False, verdict="fail",
                              error_signature="ImportError: no module", diff="BROKEN_DIFF_HERE"))
    op, pre = decide_operator(b)
    assert op == "debug"
    assert "BROKEN_DIFF_HERE" in pre        # 失败那轮的代码也带上,让 debug 定向修


def test_diff_truncated(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    monkeypatch.setenv("AUTORESEARCH_OPERATOR_DIFF_MAX", "250")
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, verdict="new_best", metric_value=1.4,
                              lower_is_better=True, diff="X" * 5000))
    _, pre = decide_operator(b)
    assert "diff 已截断" in pre and pre.count("X") <= 300


def test_prior_worktree_matches_the_best_improve_card(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import (
        ExperienceCard, prior_worktree_for_next_operator)
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=1.2,
                              lower_is_better=True, repo_path="/work/repo-2"))
    b.add_card(ExperienceCard(round=2, valid=True, metric_value=0.9,
                              lower_is_better=True, repo_path="/work/repo-3"))
    b.add_card(ExperienceCard(round=3, valid=True, metric_value=1.0,
                              lower_is_better=True, repo_path="/work/repo-4"))
    assert prior_worktree_for_next_operator(b) == "/work/repo-3"


def test_prior_worktree_prefers_recovered_tree_for_same_best_metric(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import (
        ExperienceCard, prior_worktree_for_next_operator)
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=0.9,
                              lower_is_better=True))
    b.add_card(ExperienceCard(round=2, valid=True, metric_value=1.0,
                              lower_is_better=True, repo_path="/work/repo-worse"))
    b.add_card(ExperienceCard(round=3, valid=True, metric_value=0.9,
                              lower_is_better=True, repo_path="/work/repo-recovered"))
    assert prior_worktree_for_next_operator(b) == "/work/repo-recovered"


def test_prior_worktree_debug_uses_failed_worktree(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import (
        ExperienceCard, prior_worktree_for_next_operator)
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=1.2,
                              repo_path="/work/repo-2"))
    b.add_card(ExperienceCard(round=2, valid=False, error_signature="build failed",
                              repo_path="/work/repo-3"))
    assert prior_worktree_for_next_operator(b) == "/work/repo-3"


def test_prior_worktree_failed_legacy_card_without_tree_falls_back_to_best(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import (
        ExperienceCard, prior_worktree_for_next_operator)
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=0.9,
                              repo_path="/work/repo-best"))
    b.add_card(ExperienceCard(round=2, valid=False, error_signature="run failed"))
    assert prior_worktree_for_next_operator(b) == "/work/repo-best"


# --- 方法反思回路(会议2026-08-05): agent 自报 changes_summary/rationale/bottleneck ---------

def test_card_roundtrips_selfreport_fields():
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard
    c = ExperienceCard(round=1, changes_summary="加了RMSNorm", rationale="怀疑norm是瓶颈",
                       bottleneck="attention仍慢")
    d = ExperienceCard.from_dict(c.to_dict())
    assert d.changes_summary == "加了RMSNorm" and d.rationale == "怀疑norm是瓶颈" and d.bottleneck == "attention仍慢"


def test_improve_carries_self_report_when_no_diff(tmp_path):
    # 远程路径常常没有 git diff -> 必须靠 agent 自报的 changes_summary/rationale/bottleneck 带方向
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, verdict="new_best", metric_value=1.4,
                              lower_is_better=True, diff="",
                              changes_summary="把batch从32提到64", rationale="梯度噪声太大",
                              bottleneck="学习率schedule还没调"))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "把batch从32提到64" in pre           # 自报改动进了提示(diff 为空也不丢)
    assert "梯度噪声太大" in pre                 # 理由
    assert "学习率schedule还没调" in pre         # 瓶颈 -> 下一轮有方向


def test_change_block_prefers_summary_and_keeps_diff(tmp_path):
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard, decide_operator
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, verdict="new_best", metric_value=0.9,
                              lower_is_better=False, diff="+extra=1",
                              changes_summary="改了X", bottleneck="Y是瓶颈"))
    _, pre = decide_operator(b)
    assert "改了X" in pre and "+extra=1" in pre and "Y是瓶颈" in pre   # 摘要+diff+瓶颈都在


# --- C-4: 反思判定方法层要改 -> summarize_for_regen 带上 method_note(接回换/改method) -------

def test_summarize_surfaces_reflection_method_note(tmp_path):
    import json as _j
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, method_family="lr", metric_value=1.4, valid=True))
    # 写同目录 reflection_reports.jsonl,最后一条 route=revise_method
    rp = tmp_path / "reflection_reports.jsonl"
    rp.write_text(
        _j.dumps({"route": "stay_experiment", "method_note": "旧的"}) + "\n" +
        _j.dumps({"route": "revise_method", "method_note": "RMSNorm假设错了"}) + "\n",
        encoding="utf-8")
    s = b.summarize_for_regen()
    assert "反思判定方法层需修订" in s and "RMSNorm假设错了" in s


def test_summarize_ignores_stay_experiment_reflection(tmp_path):
    import json as _j
    from autoresearch.experiments.repo_improve.search_memory import ExperienceCard
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, method_family="lr", metric_value=1.4, valid=True))
    (tmp_path / "reflection_reports.jsonl").write_text(
        _j.dumps({"route": "stay_experiment", "method_note": "n"}) + "\n", encoding="utf-8")
    assert "反思判定方法层需修订" not in b.summarize_for_regen()   # 只在 revise_method 才带


# ---- 会议 2026-08-11: 实验层消融钩子 check_expectation + crossover 失活 ----
from autoresearch.experiments.repo_improve.search_memory import check_expectation


def test_check_expectation_all_met_routes_improve():
    r = check_expectation([{
        "component": "opt", "type": "gain", "actual": -0.03,
        "expected": {"operator": "<=", "value": -0.02},
        "evidence": {"source": "ablation", "ref": "run:1"}, "verdict": "met",
    }])
    assert r["route"] == "improve" and r["failing"] == []


def test_check_expectation_partial_revises_component_when_below_declared_expectation():
    r = check_expectation([{
        "component": "opt", "type": "gain", "actual": -0.001,
        "expected": {"operator": "<=", "value": -0.02},
        "evidence": {"source": "ablation", "ref": "run:1"}, "verdict": "partial",
    }])
    assert r["route"] == "revise_method" and r["failing"] == ["opt"]


def test_check_expectation_unmet_routes_revise_with_component():
    r = check_expectation([
        {"component": "opt", "type": "gain", "actual": -0.03,
         "expected": {"operator": "<=", "value": -0.02},
         "evidence": {"source": "ablation", "ref": "run:1"}, "verdict": "met"},
        {"component": "decouple", "type": "functional", "actual": 0.4,
         "expected": {"operator": "<=", "value": 0.1},
         "evidence": {"source": "ablation", "ref": "run:1"}, "verdict": "unmet"}])
    assert r["route"] == "revise_method"
    assert r["failing"] == ["decouple"]


def test_check_expectation_low_confidence_does_not_block_decision3():
    # 决定3: 判据缺失/测不了 -> 不阻塞, 只标记
    r = check_expectation([{"component": "x", "type": "functional", "verdict": "low_confidence"}])
    assert r["route"] == "improve"
    assert r["low_confidence"] == ["x"]
    assert "全部符合预期" not in r["summary"]
    assert "不能声称符合" in r["summary"]


def test_check_expectation_unknown_verdict_is_low_confidence_not_met():
    r = check_expectation([{"component": "x", "type": "gain", "verdict": "maybe"}])
    assert r["route"] == "improve"
    assert r["low_confidence"] == ["x"]
    assert "x" not in r["met"]


def test_check_expectation_met_without_actual_or_expected_is_low_confidence():
    r = check_expectation([{"component": "x", "type": "gain", "verdict": "met"}])
    assert r["route"] == "improve"
    assert r["low_confidence"] == ["x"]


def test_check_expectation_rejects_untraceable_evidence_object():
    r = check_expectation([{
        "component": "x", "type": "gain", "actual": 0.5,
        "expected": {"operator": ">", "value": 0.1},
        "evidence": {"claim": "sandbox says so"}, "verdict": "met",
    }])
    assert r["met"] == []
    assert r["low_confidence"] == ["x"]


def test_check_expectation_recomputes_lie_from_structured_evidence():
    """A sandbox `met` label cannot override an observed value that misses its threshold."""
    r = check_expectation([{
        "component": "router",
        "type": "functional",
        "actual": 0.4,
        "expected": {"operator": "<=", "value": 0.1},
        "evidence": {"source": "run_record", "ref": "artifact://router-check"},
        "verdict": "met",
    }])

    assert r["route"] == "revise_method"
    assert r["failing"] == ["router"]
    assert r["met"] == []


def test_check_expectation_reports_component_types_for_method_level_routing():
    r = check_expectation([
        {"component": "adapter", "type": "functional", "actual": "corr=0.04",
         "expected": "corr<0.1", "verdict": "met"},
        {"component": "optimizer", "type": "gain", "actual": "-0.03 bpb",
         "expected": "lower bpb", "verdict": "met"},
    ])
    assert r["gain_components"] == ["optimizer"]
    assert r["functional_components"] == ["adapter"]


def test_check_expectation_empty_report_routes_improve():
    assert check_expectation([])["route"] == "improve"


def test_crossover_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_ENABLE_CROSSOVER", raising=False)
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=True, metric_value=1.0, lower_is_better=True))
    b.add_card(ExperienceCard(round=2, valid=True, metric_value=0.9, lower_is_better=True))
    op, _ = decide_operator(b)
    assert op == "improve"   # 默认不再走 crossover
    monkeypatch.setenv("AUTORESEARCH_ENABLE_CROSSOVER", "1")
    assert decide_operator(b)[0] == "improve"


def test_decide_operator_grounded_partial_implementation_continues_with_improve(tmp_path):
    b = SearchBoard(str(tmp_path / "memory.json"))
    b.add_card(ExperienceCard(
        round=1,
        operator="draft",
        metric_value=1.2,
        valid=True,
        verdict="valid_not_best",
        implementation_verdict="partial",
    ))

    operator, prompt = decide_operator(b)

    assert operator == "improve"
    assert "下一个尚未实现的组件" in prompt


def test_decide_operator_drifted_measurement_is_retained_and_realigned(tmp_path):
    b = SearchBoard(str(tmp_path / "memory.json"))
    worktree = str(tmp_path / "working-variant")
    b.add_card(ExperienceCard(
        round=1,
        operator="draft",
        metric_value=0.9,
        valid=True,
        verdict="valid_not_best",
        implementation_verdict="drifted",
        changes_summary="gradient clipping improved the metric",
        repo_path=worktree,
    ))

    operator, prompt = decide_operator(b)

    assert operator == "improve"
    assert "off-method observation" in prompt
    assert "确认方法靠拢" in prompt
    assert "正式 RevisingMethod" in prompt
    assert prior_worktree_for_next_operator(b) == worktree


def test_decide_operator_valid_not_best_goes_check_expectation(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, operator="draft", valid=True, verdict="valid_not_best",
                              metric_value=1.6, lower_is_better=True))
    op, pre = decide_operator(b)
    assert op == "check-expectation"
    assert "ABLATION_REPORT" in pre
    assert "每个已声明组件" in pre and "不得只挑" in pre


def test_decide_operator_invalid_non_crash_goes_check_expectation(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, operator="draft", valid=False, error_signature="",
                              verdict="invalid", lower_is_better=True))
    op, _ = decide_operator(b)
    assert op == "check-expectation"


def test_decide_operator_crash_still_debug_not_check(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, valid=False, error_signature="RuntimeError: boom"))
    assert decide_operator(b)[0] == "debug"


def test_decide_operator_new_best_goes_improve_not_check(tmp_path):
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, operator="draft", valid=True, verdict="new_best",
                              metric_value=1.4, lower_is_better=True, diff="+lr=3e-4"))
    op, pre = decide_operator(b)
    assert op == "improve"
    assert "下一个尚未实现的组件" in pre


def test_decide_operator_no_reentry_after_check_expectation(tmp_path):
    # 防重入: 上一轮已是 check-expectation 且没涨点, 不再连着查
    b = _board(tmp_path)
    b.add_card(ExperienceCard(round=1, operator="check-expectation", valid=True,
                              verdict="valid_not_best", metric_value=1.6, lower_is_better=True))
    assert decide_operator(b)[0] != "check-expectation"
