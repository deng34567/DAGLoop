"""Guard the paper-reproduction routing predicate: a github repo task must NEVER be
mistaken for a paper task, and a paper/arXiv reproduction request must be recognized."""
from __future__ import annotations

import pytest

from autoresearch.agents.runner_agent.handler import _is_paper_repro_task


@pytest.mark.parametrize("question,expected", [
    ("Reproduce the MNIST CNN baseline from the paper", True),
    ("reproduce arxiv:1512.03385 ResNet baseline", True),
    ("Reproduce the baseline reported in https://arxiv.org/abs/1706.03762", True),
    ("Clone and improve https://github.com/karpathy/nanoGPT", False),   # repo URL wins
    ("reproduce the results of github.com/foo/bar's paper", False),      # repo URL wins
    ("lower val loss on shakespeare_char", False),
    ("Improve the CIFAR-10 airbench speedrun", False),
])
def test_is_paper_repro_task(question, expected):
    assert _is_paper_repro_task(question) is expected
