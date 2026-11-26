"""
Tests for MossObservationStore — MOSS's cross-session signal/L3-alert store.
MOSS 跨会话信号/L3 告警存储的测试。

No Redis / network / EventBus / concurrency — a plain tmp sqlite file. The
persistence test (close + reopen at the SAME path) is the key property: this
store's whole point is surviving across sessions, unlike an in-memory store.
"""

from __future__ import annotations

import json

from autoresearch.agents.moss_agent.observation_store import MossObservationStore


def test_distinct_sessions_and_signal_count(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    store.record_signal("s1", "CoderAgent", "forbidden_file")
    store.record_signal("s1", "CoderAgent", "forbidden_file")
    store.record_signal("s2", "CoderAgent", "forbidden_file")

    assert store.distinct_sessions("CoderAgent", "forbidden_file") == 2
    assert store.signal_count("CoderAgent", "forbidden_file") == 3
    store.close()


def test_unseen_agent_reason_is_zero(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    assert store.distinct_sessions("NoSuchAgent", "no_such_reason") == 0
    assert store.signal_count("NoSuchAgent", "no_such_reason") == 0
    store.close()


def test_record_signal_detail_and_task_mode_round_trip(tmp_path):
    db_path = tmp_path / "moss.db"
    store = MossObservationStore(db_path)
    store.record_signal(
        "s3", "CoderAgent", "forbidden_file", detail={"k": "v"}, task_mode="theory"
    )

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT session_id, agent, reason, task_mode, detail_json FROM signals"
    ).fetchone()
    conn.close()

    assert row[0] == "s3"
    assert row[1] == "CoderAgent"
    assert row[2] == "forbidden_file"
    assert row[3] == "theory"
    assert json.loads(row[4]) == {"k": "v"}
    store.close()


def test_flag_l3_and_list_alerts(tmp_path):
    store = MossObservationStore(tmp_path / "moss.db")
    store.flag_l3(
        "s5",
        "recurring_pattern",
        {"agent": "CoderAgent", "reason": "forbidden_file", "distinct_sessions": 5},
    )

    alerts = store.list_alerts()
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["session_id"] == "s5"
    assert alert["alert_kind"] == "recurring_pattern"
    assert alert["detail"] == {
        "agent": "CoderAgent",
        "reason": "forbidden_file",
        "distinct_sessions": 5,
    }
    assert "created_at" in alert
    assert "id" in alert
    store.close()


def test_persistence_across_reopen_same_path(tmp_path):
    db_path = tmp_path / "moss.db"

    store1 = MossObservationStore(db_path)
    store1.record_signal("s1", "CoderAgent", "forbidden_file")
    store1.close()

    store2 = MossObservationStore(db_path)
    assert store2.signal_count("CoderAgent", "forbidden_file") > 0
    store2.close()
