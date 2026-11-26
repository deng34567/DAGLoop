"""Unit tests for autoresearch.eventbus.broker.dispatch."""
from __future__ import annotations

import pytest

from autoresearch.eventbus.broker.dispatch import Dispatcher
from autoresearch.eventbus.broker.envelope import Envelope


def _env(event_type: str) -> Envelope:
    return Envelope(
        event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        event_type=event_type,
        emitter="CoderAgent",
        timestamp="2026-05-23T12:00:00Z",
        schema_version=1,
        cost_delta="0",
        payload={},
    )


async def test_exact_pattern_matches():
    d = Dispatcher()
    seen: list[str] = []

    async def h(env):
        seen.append(env.event_type)

    d.register("coder.module_implemented", h)
    await d.dispatch(_env("coder.module_implemented"))
    assert seen == ["coder.module_implemented"]


async def test_exact_pattern_does_not_match_other_event():
    d = Dispatcher()
    seen: list[str] = []

    async def h(env):
        seen.append(env.event_type)

    d.register("coder.module_implemented", h)
    await d.dispatch(_env("coder.module_fixed"))
    assert seen == []


async def test_wildcard_matches_all():
    d = Dispatcher()
    seen: list[str] = []

    async def h(env):
        seen.append(env.event_type)

    d.register("*", h)
    for et in (
        "coder.module_implemented",
        "critic.review_completed",
        "moss.intervention_completed",
    ):
        await d.dispatch(_env(et))
    assert len(seen) == 3


async def test_prefix_glob_pattern_filters_correctly():
    d = Dispatcher()
    matched: list[str] = []

    async def h(env):
        matched.append(env.event_type)

    d.register("coder.*_requested", h)

    # matches
    await d.dispatch(_env("coder.module_implementation_requested"))
    await d.dispatch(_env("coder.fix_apply_requested"))
    # does NOT match (critic, not coder)
    await d.dispatch(_env("critic.review_completed"))
    # does NOT match (coder but not *_requested suffix)
    await d.dispatch(_env("coder.module_implemented"))

    assert matched == [
        "coder.module_implementation_requested",
        "coder.fix_apply_requested",
    ]


async def test_multiple_patterns_all_match_event_calls_all_handlers():
    d = Dispatcher()
    calls: list[str] = []

    async def h_specific(env):
        calls.append("specific")

    async def h_prefix(env):
        calls.append("prefix")

    async def h_wildcard(env):
        calls.append("wildcard")

    d.register("coder.module_implemented", h_specific)
    d.register("coder.*", h_prefix)
    d.register("*", h_wildcard)

    await d.dispatch(_env("coder.module_implemented"))
    # All three should fire (order = registration order, sequential)
    assert calls == ["specific", "prefix", "wildcard"]


async def test_handler_exception_prevents_successful_acknowledgement():
    d = Dispatcher()
    calls: list[str] = []

    async def h_bad(env):
        calls.append("bad-before-raise")
        raise RuntimeError("boom")

    async def h_good(env):
        calls.append("good")

    d.register("coder.module_implemented", h_bad)
    d.register("coder.module_implemented", h_good)

    with pytest.raises(RuntimeError, match="boom"):
        await d.dispatch(_env("coder.module_implemented"))
    assert calls == ["bad-before-raise"]


async def test_no_matching_handler_is_no_op():
    d = Dispatcher()
    async def h(env):
        raise AssertionError("should not be called")
    d.register("critic.review_completed", h)

    # different event type -> no handler matches -> no error
    await d.dispatch(_env("coder.module_implemented"))


async def test_matching_handlers_returns_list():
    d = Dispatcher()

    async def h1(env): ...
    async def h2(env): ...

    d.register("coder.*", h1)
    d.register("*", h2)

    matched = d.matching_handlers("coder.module_implemented")
    assert h1 in matched
    assert h2 in matched
    assert len(matched) == 2
