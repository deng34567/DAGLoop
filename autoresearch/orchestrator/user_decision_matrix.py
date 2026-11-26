"""User-decision reason-matrix loader + UserDecisionGuard.

Loads the existing SSOT YAML
(state-machine/user-decisions/reason-matrix.yaml, V1.10 §3) and exposes a small
guard object for looking up legal actions per ``awaiting_user_decision_reason``,
their ``next_state`` / ``side_effects``. Lives in the orchestrator package (not
agents/) so the engine imports it directly, same as moss_matrix.py.

Mirrors moss_matrix.py's yaml.safe_load + guard style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

DEFAULT_MATRIX_PATH = (
    Path(__file__).parent / "state-machine" / "user-decisions" / "reason-matrix.yaml"
)


def load_user_decision_matrix(
    path: Union[str, Path, None] = None
) -> dict[str, dict[str, dict]]:
    """Load and parse the user-decision reason-matrix YAML.

    Returns ``{reason: {action: {"next_state": str, "side_effects": [str]}}}``.
    """
    matrix_path = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    data = yaml.safe_load(matrix_path.read_text())

    matrix: dict[str, dict[str, dict]] = {}
    for entry in data.get("reasons", []):
        reason = entry["reason"]
        by_action: dict[str, dict] = {}
        for a in entry.get("actions", []):
            by_action[a["action"]] = {
                "next_state": a.get("next_state"),
                "side_effects": a.get("side_effects", []),
            }
        matrix[reason] = by_action
    return matrix


class UserDecisionGuard:
    """Validates and looks up user-decisions against the loaded reason-matrix."""

    def __init__(self, matrix: dict[str, dict[str, dict]]) -> None:
        self._matrix = matrix

    def legal_actions(self, reason: str) -> list[str]:
        return list(self._matrix.get(reason, {}).keys())

    def is_legal(self, reason: str, action: str) -> bool:
        return action in self._matrix.get(reason, {})

    def next_state(self, reason: str, action: str) -> str:
        return self._matrix[reason][action]["next_state"]

    def side_effects(self, reason: str, action: str) -> list[str]:
        return self._matrix[reason][action].get("side_effects", [])
