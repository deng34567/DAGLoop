"""Import-smoke test for the ported _adapter modules (Task 8 / M0).

These modules (BrokerBus, MeteredProvider, CostMeter) are ported from vibe
for LATER agent use; M0 only requires that they import cleanly and expose
their expected public API. No behavior is exercised here and no LLM is
called.
"""

from __future__ import annotations


def test_cost_meter_imports():
    from autoresearch.agents._adapter.cost_meter import CostMeter

    assert CostMeter is not None
    meter = CostMeter()
    assert hasattr(meter, "add")
    assert hasattr(meter, "snapshot_reset")


def test_broker_bus_imports():
    from autoresearch.agents._adapter.broker_bus import BrokerBus, causation_var

    assert BrokerBus is not None
    assert hasattr(BrokerBus, "subscribe")
    assert hasattr(BrokerBus, "publish")
    assert causation_var is not None


def test_llm_provider_imports():
    from autoresearch.agents._adapter.llm_provider import MeteredProvider

    assert MeteredProvider is not None
    assert hasattr(MeteredProvider, "call")


def test_adapter_package_public_api():
    from autoresearch.agents._adapter import BrokerBus, CostMeter, MeteredProvider, causation_var

    assert {BrokerBus.__name__, CostMeter.__name__, MeteredProvider.__name__} == {
        "BrokerBus",
        "CostMeter",
        "MeteredProvider",
    }
    assert causation_var is not None
