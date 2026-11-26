"""Fix C P1: the global LLM concurrency cap (overheating guard for concurrent sessions)."""
from __future__ import annotations

import threading
import time

from autoresearch.llm import client as C


def test_llm_slot_never_exceeds_cap(monkeypatch):
    # more threads than the cap must never run more than `cap` completions at once.
    cap = 3
    monkeypatch.setattr(C, "_LLM_SEMAPHORE", threading.BoundedSemaphore(cap))
    seen = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def work():
        with C.llm_concurrency_slot():
            with lock:
                seen["cur"] += 1
                seen["max"] = max(seen["max"], seen["cur"])
            time.sleep(0.05)
            with lock:
                seen["cur"] -= 1

    threads = [threading.Thread(target=work) for _ in range(cap + 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen["max"] <= cap            # the invariant: never more than cap in flight
    assert seen["max"] >= 1              # and work actually ran


def test_llm_slot_releases_on_exception(monkeypatch):
    # a failing completion must release its slot (else the pool leaks and eventually deadlocks).
    monkeypatch.setattr(C, "_LLM_SEMAPHORE", threading.BoundedSemaphore(1))
    for _ in range(3):
        try:
            with C.llm_concurrency_slot():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    # if the slot leaked, this acquire would block forever; guard with a timeout.
    assert C._LLM_SEMAPHORE.acquire(timeout=1) is True
    C._LLM_SEMAPHORE.release()
