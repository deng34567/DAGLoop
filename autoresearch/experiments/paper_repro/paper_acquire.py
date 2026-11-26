"""Acquire the doc2json-format paper JSON that Paper2Code's pipeline consumes.

Paper2Code's ``0_pdf_process`` expects a science-parse / doc2json JSON
(``{title, abstract, pdf_parse:{body_text:[{text,section,sec_num}], ...}}``) that
is normally produced by running a PDF parser over the paper's PDF. We have no PDF
parser wired, so we synthesize that JSON with the LLM from the research question
(which names/describes the paper to reproduce) plus the model's own knowledge of
the paper. This is enough to reproduce a well-established BASELINE: Paper2Code
turns the JSON into code and the agentic loop runs it to the reported number.

Kept deliberately small and offline-testable: ``acquire_paper_json(question, llm,
...)`` takes any object with a ``.chat(messages=..., response_format=...)`` method
(our LLMClient or a fake) and writes the cleaned JSON, returning its path.
"""
from __future__ import annotations

import json
import os
import re

from autoresearch.agents._reuse.json_extract import extract_json

_SYSTEM = (
    "You are a careful research engineer preparing a paper for faithful REPRODUCTION of its "
    "reported BASELINE result. Given a request that names or describes a paper, produce a compact "
    "structured description with ENOUGH concrete detail to re-implement the baseline from scratch: "
    "exact model architecture (layers, sizes), dataset and preprocessing, optimizer + full training "
    "schedule (lr, batch size, epochs/steps, scheduler), the PRIMARY evaluation metric, and the "
    "paper's REPORTED baseline value for that metric. Use the paper's real, well-known numbers; do "
    "not invent an implausible baseline. Return ONLY JSON, no prose."
)

_USER_TMPL = (
    "Request:\n{question}\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    '  "title": string  -- the paper title (include the reported baseline metric+value if known)\n'
    '  "abstract": string  -- 2-4 sentences; state the reported baseline value explicitly\n'
    '  "metric": string  -- the primary metric name (e.g. "test_accuracy", "val_loss", "BLEU")\n'
    '  "lower_is_better": boolean\n'
    '  "baseline_value": number  -- the paper\'s reported baseline value for that metric\n'
    '  "sections": [ {{"section": string, "text": string}} ]  -- 4-7 sections covering '
    "Introduction, Dataset, Model Architecture, Training, and Evaluation/Baseline; each 'text' must "
    "be detailed enough to re-implement (concrete numbers, not vague prose). The Evaluation section "
    "MUST state the reported baseline value and the acceptable reproduction range.\n"
)


def _slug(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", (title or "Paper").title())[:40]
    return s or "Paper"


def acquire_paper_json(question: str, llm, *, workdir: str, paper_name: str | None = None) -> tuple[str, dict]:
    """Synthesize the doc2json paper JSON for ``question`` and write it under ``workdir``.

    Returns ``(json_path, meta)`` where ``meta`` carries ``{paper_name, metric,
    lower_is_better, baseline_value, title}`` for the caller (hints for the agentic run + the
    reproduction check). Raises on an unparseable LLM response -- the caller degrades to a hard
    fail (there is no paper to build from)."""
    result = llm.chat(
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": _USER_TMPL.format(question=question)}],
        response_format="json_object")
    obj = json.loads(extract_json(result.content or "{}"))

    title = str(obj.get("title") or "Reproduced Paper").strip()
    name = paper_name or _slug(title)
    abstract = str(obj.get("abstract") or "").strip()
    metric = str(obj.get("metric") or "").strip() or None
    lower = bool(obj.get("lower_is_better", False))
    baseline_value = obj.get("baseline_value")
    sections = obj.get("sections") or []

    body_text = []
    for i, sec in enumerate(sections, start=1):
        if not isinstance(sec, dict):
            continue
        body_text.append({"text": str(sec.get("text") or "").strip(),
                          "section": str(sec.get("section") or f"Section {i}").strip(),
                          "sec_num": str(i)})
    if not body_text:  # never hand Paper2Code an empty paper -- fold the abstract in as one section
        body_text = [{"text": abstract or title, "section": "Summary", "sec_num": "1"}]

    cleaned = {
        "paper_id": name,
        "title": title,
        "abstract": abstract or title,
        "pdf_parse": {
            "paper_id": name,
            "abstract": [{"text": abstract or title, "section": "Abstract"}],
            "body_text": body_text,
            "back_matter": [],
            "ref_entries": {},
        },
    }
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"{name}_cleaned.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    meta = {"paper_name": name, "metric": metric, "lower_is_better": lower,
            "baseline_value": baseline_value, "title": title}
    return path, meta


__all__ = ["acquire_paper_json"]
