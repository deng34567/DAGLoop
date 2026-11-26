"""自动挑一张真正空闲的卡:靠探测,绝不写死卡号。"""
from __future__ import annotations

from autoresearch.experiments.repo_improve.gpu_pick import pick_free_card, _choose


def _row(idx, used, total=46068, util=0):
    return {"index": str(idx), "used": used, "total": total, "util": util}


def test_prefers_the_card_with_most_free_memory_and_idle():
    rows = [_row(0, 15000), _row(1, 40000), _row(2, 200), _row(3, 19000)]
    assert _choose(rows, min_free_mib=8000) == "2"


def test_busy_cards_are_skipped_even_if_memory_looks_free():
    # 显存空但在满载计算 -> 不抢,免得两个训练互相拖慢
    rows = [_row(0, 200, util=100), _row(1, 5000, util=0)]
    assert _choose(rows, min_free_mib=1000) == "1"


def test_cards_without_enough_free_memory_are_skipped():
    rows = [_row(0, 44000), _row(1, 43000)]      # 都只剩 2-3G
    assert _choose(rows, min_free_mib=8000) is None


def test_no_gpu_at_all_means_cpu():
    assert _choose([], min_free_mib=1000) is None


def test_explicit_env_override_wins(monkeypatch):
    # 想钉死某张卡时仍可覆盖(运维需要),但默认是自动
    monkeypatch.setenv("AUTORESEARCH_REPO_IMPROVE_CUDA", "5")
    assert pick_free_card() == "5"


def test_probe_failure_degrades_to_cpu(monkeypatch):
    # 没有 nvidia-smi / 探测失败 -> 返回 None 走 CPU,绝不把流程带崩
    monkeypatch.delenv("AUTORESEARCH_REPO_IMPROVE_CUDA", raising=False)
    monkeypatch.setattr("autoresearch.experiments.repo_improve.gpu_pick._probe",
                        lambda: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert pick_free_card() is None
