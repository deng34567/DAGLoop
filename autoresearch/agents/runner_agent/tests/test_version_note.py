"""#9(会议 2026-08-05, 老师): new_best 时记录"当前版本特点"到 version_notes.md。"""
import autoresearch.experiments.repo_improve.search_memory as sm
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler


def test_persist_version_note_writes_on_new_best(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "session_workroot", lambda sid: str(tmp_path))
    h = RunnerAgentHandler.__new__(RunnerAgentHandler)   # 裸实例, 该方法不碰 self 属性
    h._persist_version_note("sid123", 2, {
        "method_metric": 1.23, "changes_summary": "加了RMSNorm",
        "rationale": "norm是瓶颈", "bottleneck": "attention仍慢"})
    txt = (tmp_path / "version_notes.md").read_text(encoding="utf-8")
    assert "Round 2" in txt and "metric=1.23" in txt
    assert "加了RMSNorm" in txt and "norm是瓶颈" in txt and "attention仍慢" in txt


def test_version_note_appends_multiple(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "session_workroot", lambda sid: str(tmp_path))
    h = RunnerAgentHandler.__new__(RunnerAgentHandler)
    h._persist_version_note("s", 1, {"method_metric": 2.0, "changes_summary": "A"})
    h._persist_version_note("s", 3, {"method_metric": 1.5, "changes_summary": "B"})
    txt = (tmp_path / "version_notes.md").read_text(encoding="utf-8")
    assert "Round 1" in txt and "Round 3" in txt and txt.index("Round 1") < txt.index("Round 3")


def test_version_note_best_effort_no_raise(tmp_path, monkeypatch):
    # session_workroot 抛错 -> 不炸(best-effort)
    def boom(sid): raise RuntimeError("x")
    monkeypatch.setattr(sm, "session_workroot", boom)
    h = RunnerAgentHandler.__new__(RunnerAgentHandler)
    h._persist_version_note("s", 1, {"method_metric": 1.0})   # 不抛
