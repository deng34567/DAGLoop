from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

from autoresearch.web import server


def test_concurrent_topic_updates_preserve_both_sessions(monkeypatch, tmp_path):
    path = tmp_path / "topics.json"
    path.write_text("{}")
    monkeypatch.setattr(server, "_WORKDIR", tmp_path)
    monkeypatch.setattr(server, "_TOPICS_FILE", path)
    original_write = Path.write_text
    first_writing = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()

    def write(self, data, *args, **kwargs):
        if self.parent == tmp_path and '"first"' in data and '"second"' not in data:
            first_writing.set()
            assert release_first.wait(5)
        return original_write(self, data, *args, **kwargs)

    def second():
        try:
            server._record_topic("second", "Second task")
        finally:
            second_done.set()

    monkeypatch.setattr(Path, "write_text", write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(server._record_topic, "first", "First task")
        assert first_writing.wait(5)
        other = pool.submit(second)
        second_done.wait(0.5)
        release_first.set()
        first.result(timeout=5)
        other.result(timeout=5)
    assert set(json.loads(path.read_text())) == {"first", "second"}


def test_topic_update_recovers_a_corrupted_display_cache(monkeypatch, tmp_path):
    path = tmp_path / "topics.json"
    path.write_text('{"truncated":')
    monkeypatch.setattr(server, "_WORKDIR", tmp_path)
    monkeypatch.setattr(server, "_TOPICS_FILE", path)
    server._record_topic("sid", "Recorded task")
    assert server._topics()["sid"]["topic"] == "Recorded task"
