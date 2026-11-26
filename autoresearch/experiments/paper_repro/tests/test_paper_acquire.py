"""Offline test for paper acquisition: a fake LLM returns a structured paper description,
and we assert the doc2json cleaned JSON that Paper2Code consumes is assembled correctly."""
from __future__ import annotations

import json

from autoresearch.experiments.paper_repro.paper_acquire import acquire_paper_json


class _Res:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        return _Res(self._content)


_PAPER = json.dumps({
    "title": "A Small CNN Baseline for MNIST (~99.2% test accuracy)",
    "abstract": "A LeNet-style CNN reaches ~99.2% test accuracy on MNIST.",
    "metric": "test_accuracy",
    "lower_is_better": False,
    "baseline_value": 99.2,
    "sections": [
        {"section": "Model Architecture", "text": "Conv(1->32)-Conv(32->64)-Pool-FC."},
        {"section": "Evaluation and Baseline", "text": "Reported baseline ~99.2% top-1 test acc."},
    ],
})


def test_acquire_writes_doc2json_cleaned(tmp_path):
    llm = _FakeLLM(_PAPER)
    path, meta = acquire_paper_json("Reproduce the MNIST CNN baseline paper", llm,
                                    workdir=str(tmp_path))
    assert path.endswith("_cleaned.json")
    obj = json.loads(open(path, encoding="utf-8").read())
    # doc2json shape Paper2Code's 0_pdf_process expects
    assert obj["title"].startswith("A Small CNN")
    assert obj["pdf_parse"]["body_text"][0]["section"] == "Model Architecture"
    assert obj["pdf_parse"]["body_text"][0]["sec_num"] == "1"
    assert obj["pdf_parse"]["body_text"][1]["section"] == "Evaluation and Baseline"
    # meta carries the reproduction hints
    assert meta["metric"] == "test_accuracy"
    assert meta["lower_is_better"] is False
    assert meta["baseline_value"] == 99.2
    assert meta["paper_name"]  # a slug was derived


def test_acquire_empty_sections_folds_abstract(tmp_path):
    content = json.dumps({"title": "T", "abstract": "the abstract", "metric": "acc",
                          "lower_is_better": False, "baseline_value": 90, "sections": []})
    path, meta = acquire_paper_json("reproduce paper T", _FakeLLM(content), workdir=str(tmp_path))
    obj = json.loads(open(path, encoding="utf-8").read())
    assert len(obj["pdf_parse"]["body_text"]) == 1
    assert obj["pdf_parse"]["body_text"][0]["text"] == "the abstract"
