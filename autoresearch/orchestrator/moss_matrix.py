"""MOSS action-matrix loader + MossActionGuard.

Loads the existing SSOT YAML
(state-machine/moss/moss-action-matrix.yaml, V1.10 §6) and exposes a small
guard object for looking up legal actions per current_moss_reason, their
next_state / counter_updates / with_reason, and a conservative fallback
action. Lives in the orchestrator package (not agents/) so both the engine
and the MOSS agent import the same module.

Mirrors the yaml.safe_load + path style of orchestrator/loader.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

DEFAULT_MATRIX_PATH = (
    Path(__file__).parent / "state-machine" / "moss" / "moss-action-matrix.yaml"
)


def load_moss_action_matrix(
    path: Union[str, Path, None] = None
) -> dict[str, dict[str, dict]]:
    """Load and parse the MOSS action matrix YAML.

    Returns ``{reason: {action: {"next_state": str, "counter_updates": [str],
    "with_reason": str | None}}}``.
    """
    matrix_path = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    data = yaml.safe_load(matrix_path.read_text())

    matrix: dict[str, dict[str, dict]] = {}
    for entry in data.get("MossActionMatrix", []):
        reason = entry["reason"]
        by_action: dict[str, dict] = {}
        for a in entry.get("actions", []):
            by_action[a["action"]] = {
                "next_state": a.get("next_state"),
                "counter_updates": a.get("counter_updates", []),
                "with_reason": a.get("with_reason"),
            }
        matrix[reason] = by_action
    return matrix


class MossActionGuard:
    """Validates and looks up MOSS actions against the loaded action matrix."""

    def __init__(self, matrix: dict[str, dict[str, dict]]) -> None:
        self._matrix = matrix

    def legal_actions(self, reason: str) -> list[str]:
        return list(self._matrix.get(reason, {}).keys())

    def is_legal(self, reason: str, action: str) -> bool:
        return action in self._matrix.get(reason, {})

    def fallback(self, reason: str) -> str:
        """The conservative action for ``reason``.

        Preference order: escalate_to_user > first request_human_* /
        human_data_review > research_rework > first legal action.
        """
        legal = self.legal_actions(reason)
        if "escalate_to_user" in legal:
            action = "escalate_to_user"
        else:
            human_actions = [
                a
                for a in legal
                if a.startswith("request_human_") or a == "human_data_review"
            ]
            if human_actions:
                action = human_actions[0]
            elif "research_rework" in legal:
                action = "research_rework"
            else:
                action = legal[0]
        assert action in legal, f"fallback action {action!r} not legal for {reason!r}"
        return action

    def next_state(self, reason: str, action: str) -> str:
        return self._matrix[reason][action]["next_state"]

    def counter_updates(self, reason: str, action: str) -> list[str]:
        return self._matrix[reason][action].get("counter_updates", [])

    def with_reason(self, reason: str, action: str) -> Optional[str]:
        return self._matrix[reason][action].get("with_reason")
