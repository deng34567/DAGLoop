"""Component DAG and crash-safe per-component execution ledger.

This is deliberately a small controller artifact, not a second state machine.
The existing experiment loop asks the ledger which component is ready, records
each real attempt, and advances only after that component's own contract passes.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any


class ComponentPlanError(ValueError):
    """The confirmed Method cannot be represented as a valid component DAG."""


class ComponentPlanFormatError(ComponentPlanError):
    """A check cannot be decoded by the numeric-evidence protocol."""


_ID_RE = re.compile(r"[^a-z0-9]+")
_THRESHOLD_RE = re.compile(r"(<=|>=|==|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _requires_repository_history_mutation(check: str) -> bool:
    """Reject component evidence recipes that conflict with frozen-HEAD capture."""
    for match in re.finditer(
            r"\bgit\s+commit\b|\b(?:advance|move|change)\s+(?:the\s+)?HEAD\b|"
            r"\bHEAD\s+(?:must\s+)?(?:advance|move|change)s?\b|"
            r"(?:HEAD\s*(?:必须|须|需要)?(?:真正)?前进|(?:推进|改变|修改)\s*HEAD)",
            str(check or ""), re.IGNORECASE):
        prefix = str(check or "")[max(0, match.start() - 40):match.start()]
        if not re.search(
                r"(?:do\s+not|don't|never|must\s+not|without|禁止|不得|不要|无需|无须|不需)"
                r"[^.;。；\n]{0,24}$", prefix, re.IGNORECASE):
            return True
    return False


def _component_id(value: Any, index: int) -> str:
    raw = str(value or "").strip().lower()
    slug = _ID_RE.sub("-", raw).strip("-")
    return slug or f"component-{index + 1}"


def method_fingerprint(method: dict) -> str:
    canonical = json.dumps(method, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _direct_gain_predicate(check: str) -> tuple[str, str, float]:
    """Decode only the supported scalar protocol, never infer a gate from prose."""
    match = re.fullmatch(
        r"\s*(signed_metric_improvement|metric delta|candidate_metric)\s*"
        + _THRESHOLD_RE.pattern + r"\s*", check)
    if match:
        measurement, operator, literal = match.groups()
        value = float(literal)
        if math.isfinite(value):
            if (measurement != "candidate_metric" and operator in {"<", "<=", "=="}
                    and value <= 0):
                raise ComponentPlanFormatError(
                    "direct-gain acceptance contradicts strict positive improvement: "
                    "signed_metric_improvement is parent - candidate when minimizing, "
                    "candidate - parent when maximizing, and must be > 0. "
                    "Formally revise the contradictory predicate; do not reverse the "
                    "frozen metric direction or rerun unchanged valid code.")
            return ("signed_metric_improvement" if measurement == "metric delta" else measurement,
                    operator, value)
    raise ComponentPlanFormatError(
        "direct-gain acceptance requires one executable comparison: "
        "signed_metric_improvement <operator> <finite number> or "
        "candidate_metric <operator> <finite number>, against the exact dependency checkpoint. "
        "Additional controls or prose conditions are unsupported; formally revise the plan "
        "without silently dropping required experiments.")


def validate_component_acceptance(plan: ComponentPlan) -> None:
    invalid = []
    explanation = ""
    for node in plan.components:
        if _requires_repository_history_mutation(node.implementation_check):
            invalid.append(
                f"{node.component_id}.implementation_check requires repository history mutation")
            explanation = (
                "Component implementation checks may not require git commit or HEAD mutation. "
                "Candidate source must remain a working-tree diff against the frozen commit so "
                "the host can capture and review the exact implementation.")
        description = (node.purpose + " " + node.mechanism).casefold()
        if (node.component_id in plan.terminal_components
                and node.role == "indirect_gain"
                and (("measurement-only" in description and "official" in description
                      and ("evaluator" in description or "test.sh" in description))
                     or ("composition/evaluation" in description
                         and "no new target-specific code" in description))
                and (re.match(r"^none\b", node.optimization_space.strip(), re.IGNORECASE)
                     or node.optimization_space.strip().casefold()
                     == "no declared optimization space")):
            invalid.append(f"{node.component_id} is only a terminal official evaluation")
            explanation = ("The unchanged official evaluator belongs in final verification, "
                           "not a separate component. Remove this node and preserve its "
                           "required check in the integrated method evaluation.")
        if (node.role == "indirect_gain"
                and re.search(r"\b(signed_metric_improvement|candidate_metric|metric delta)\s*"
                              + _THRESHOLD_RE.pattern, node.acceptance_check)):
            invalid.append(f"{node.component_id}.functional_check uses a task-performance quantity")
            explanation = ("Task-performance comparisons require direct_gain and a host-bound "
                           "parent/candidate comparison; an arbitrary raw scalar cannot prove improvement.")
        if node.role == "direct_gain":
            try:
                _direct_gain_predicate(node.acceptance_check)
            except ComponentPlanFormatError as exc:
                invalid.append(f"{node.component_id}.ablation_test={node.acceptance_check!r}")
                explanation = str(exc)
    if invalid:
        raise ComponentPlanFormatError("; ".join(invalid) + ": " + explanation)


@dataclass(frozen=True)
class ComponentNode:
    component_id: str
    name: str
    role: str
    depends_on: tuple[str, ...]
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    acceptance_requires: tuple[str, ...]
    mechanism: str
    expected_effect: str
    acceptance_check: str
    implementation_check: str
    optimization_space: str
    downstream_contract: str
    threshold_operator: str = ""
    threshold_value: float | None = None
    measurement_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # This is derived from acceptance_check, not a new persisted contract.
        data.pop("measurement_name")
        data["depends_on"] = list(self.depends_on)
        data["inputs"] = list(self.inputs)
        data["outputs"] = list(self.outputs)
        data["acceptance_requires"] = list(self.acceptance_requires)
        return data


@dataclass(frozen=True)
class ComponentPlan:
    method_fingerprint: str
    components: tuple[ComponentNode, ...]
    topological_order: tuple[str, ...]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "method_fingerprint": self.method_fingerprint,
            "components": [component.to_dict() for component in self.components],
            "topological_order": list(self.topological_order),
        }

    def node(self, component_id: str) -> ComponentNode:
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(component_id)

    def dependency_closure(self, component_id: str) -> tuple[str, ...]:
        """All transitive prerequisites in executable topological order."""
        required: set[str] = set()

        def visit(current: str) -> None:
            for dependency in self.node(current).depends_on:
                if dependency not in required:
                    required.add(dependency)
                    visit(dependency)

        visit(component_id)
        return tuple(item for item in self.topological_order if item in required)

    @property
    def terminal_components(self) -> tuple[str, ...]:
        parents = {
            dependency
            for component in self.components
            for dependency in component.depends_on
        }
        return tuple(
            component_id for component_id in self.topological_order
            if component_id not in parents
        )


def build_component_plan(method: dict, *, require_design_fields: bool = True) -> ComponentPlan:
    raw_components = method.get("components") if isinstance(method, dict) else None
    if not isinstance(raw_components, list) or not raw_components:
        raise ComponentPlanError("confirmed Method must declare at least one component")

    nodes: list[ComponentNode] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, dict):
            raise ComponentPlanError(f"components[{index}] must be an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ComponentPlanError(f"components[{index}].name is required")
        component_id = _component_id(raw.get("component_id") or name, index)
        if component_id in ids:
            raise ComponentPlanError(f"duplicate component_id: {component_id}")
        ids.add(component_id)
        role = str(raw.get("type") or "direct_gain").strip().lower()
        role = {"gain": "direct_gain", "functional": "indirect_gain"}.get(role, role)
        if role not in {"direct_gain", "indirect_gain"}:
            raise ComponentPlanError(
                f"{component_id}.type must be direct_gain or indirect_gain")
        purpose = str(raw.get("purpose") or "").strip()
        inputs_raw = raw.get("inputs") or []
        outputs_raw = raw.get("outputs") or []
        acceptance_requires_raw = raw.get("acceptance_requires") or []
        if require_design_fields and not purpose:
            raise ComponentPlanError(f"{component_id}.purpose is required")
        if not isinstance(inputs_raw, list) or not all(
            isinstance(item, str) and item.strip() for item in inputs_raw
        ):
            raise ComponentPlanError(f"{component_id}.inputs must be a non-empty string array")
        if not isinstance(outputs_raw, list) or not all(
            isinstance(item, str) and item.strip() for item in outputs_raw
        ):
            raise ComponentPlanError(f"{component_id}.outputs must be a non-empty string array")
        if require_design_fields and not inputs_raw:
            raise ComponentPlanError(f"{component_id}.inputs is required")
        if require_design_fields and not outputs_raw:
            raise ComponentPlanError(f"{component_id}.outputs is required")
        if (not isinstance(acceptance_requires_raw, list) or not all(
                isinstance(item, str) and item.strip() for item in acceptance_requires_raw)):
            raise ComponentPlanError(
                f"{component_id}.acceptance_requires must be an array of component IDs")
        mechanism = str(raw.get("mechanism") or "").strip()
        purpose = purpose or mechanism or name
        inputs_raw = inputs_raw or ["parent checkpoint"]
        outputs_raw = outputs_raw or [f"{name} output"]
        expected_effect = str(raw.get("expected_effect") or "").strip()
        if not mechanism or not expected_effect:
            raise ComponentPlanError(f"{component_id} needs mechanism and expected_effect")
        raw_check = (
            raw.get("ablation_test") if role == "direct_gain" else raw.get("functional_check"))
        acceptance_check = str(raw_check or "").strip()
        if not acceptance_check:
            check_name = "ablation_test" if role == "direct_gain" else "functional_check"
            raise ComponentPlanError(f"{component_id}.{check_name} is required")
        threshold_operator = ""
        threshold_value = None
        measurement_name = ""
        if role == "indirect_gain":
            threshold_source = acceptance_check
            structured_check = raw_check
            if isinstance(raw_check, str) and raw_check.lstrip().startswith("{"):
                # Deliberation historically stringifies dict-valued checks. Decode
                # data only; never execute the procedure or its comparison text.
                try:
                    structured_check = json.loads(raw_check)
                except ValueError:
                    try:
                        structured_check = ast.literal_eval(raw_check)
                    except (ValueError, SyntaxError):
                        structured_check = None
            if isinstance(structured_check, dict) and "numeric_comparison" in structured_check:
                threshold_source = structured_check["numeric_comparison"]
                if not isinstance(threshold_source, str):
                    raise ComponentPlanFormatError(
                        f"{component_id}.numeric_comparison must be a numeric comparison string")
            threshold_matches = _THRESHOLD_RE.findall(threshold_source)
            # A named gate may define its aggregate using several sub-comparisons.
            # Accept an explicit gate followed by its definition, including
            # "where:" with conditional definitions of that same gate. Otherwise
            # retain the rejection of ambiguous independent thresholds.
            aggregate = re.match(
                r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?P<gate>"
                + _THRESHOLD_RE.pattern
                + r")\s+where(?:\s+(?P=name)\s*=(?!=)"
                + r"|\s*:\s*(?=[\s\S]*?\b(?P=name)\s*=(?!=)))",
                threshold_source)
            if aggregate:
                measurement_name = aggregate.group("name")
                threshold_matches = _THRESHOLD_RE.findall(aggregate.group("gate"))
            else:
                # Prose may describe sub-assertions before defining one aggregate
                # and explicitly requiring it. Do not count those assertions as gates.
                requirements = list(re.finditer(
                    r"(?:\bRequire\b|要求)\s+([A-Za-z_]\w*)\s*"
                    + _THRESHOLD_RE.pattern + r"(?=\s*(?:[.;。](?=\s|$)|$))",
                    threshold_source, re.IGNORECASE))
                if len(requirements) == 1:
                    requirement = requirements[0]
                    name, operator, value = requirement.groups()
                    if re.search(r"\b" + re.escape(name) + r"\s*=(?!=)",
                                 threshold_source[:requirement.start()]):
                        threshold_matches = [(operator, value)]
                        measurement_name = name
            if len(threshold_matches) != 1:
                # An explicitly labeled gate is separate from its computation recipe.
                # Keep one gate and require the recipe to define that same scalar.
                labeled = re.fullmatch(
                    r"\s*Acceptance contract:\s*(?P<name>[A-Za-z_]\w*)\s*(?P<gate>"
                    + _THRESHOLD_RE.pattern
                    + r")\s*[.;]\s*Computation recipe:\s*(?P<recipe>[\s\S]+)",
                    threshold_source, re.IGNORECASE)
                if (labeled and len(re.findall(r"\bAcceptance contract:", threshold_source,
                                               re.IGNORECASE)) == 1
                        and re.search(r"\b" + re.escape(labeled.group("name")) + r"\s*=(?!=)",
                                      labeled.group("recipe"))):
                    threshold_matches = _THRESHOLD_RE.findall(labeled.group("gate"))
                    measurement_name = labeled.group("name")
            if not threshold_matches:
                raise ComponentPlanFormatError(
                    f"{component_id}.functional_check needs a numeric threshold: "
                    "the right-hand side must be a finite numeric literal, not a variable or file path. "
                    "For a measured reference, compute and emit difference = observed - reference "
                    "in the validation script, preserve the original comparison operator, and compare "
                    "difference against 0. Retain evidence for both operands and their prescribed source; "
                    "do not substitute an arbitrary fixed threshold or change the acceptance meaning.")
            if len(threshold_matches) != 1:
                raise ComponentPlanFormatError(
                    f"{component_id}.functional_check must expose one aggregate scalar threshold. "
                    "Set functional_check to only the named numeric comparison (for example "
                    "failed_check_count == 0). Move its complete computation recipe, all "
                    "sub-assertions and conditional branches into implementation_check; do not "
                    "discard any condition or select the first threshold from a conjunction.")
            scalar_gates = re.finditer(
                r"\b[A-Za-z_]\w*\s*" + _THRESHOLD_RE.pattern,
                threshold_source)
            if not any(
                not threshold_source[:gate.start()].rstrip().endswith(
                    ("+", "-", "*", "/", "%", "."))
                for gate in scalar_gates
            ):
                raise ComponentPlanFormatError(
                    f"{component_id}.functional_check must compare a named measured scalar "
                    "with a numeric literal; compute a variable-dependent formula in the "
                    "validation script and compare its difference or failure count with 0.")
            simple_gate = re.match(r"^\s*[A-Za-z_]\w*\s*" + _THRESHOLD_RE.pattern,
                                   threshold_source)
            if (simple_gate and re.match(r"\s*[+*/-]\s*(?:[A-Za-z_(]|\d)",
                                         threshold_source[simple_gate.end():])):
                raise ComponentPlanFormatError(
                    f"{component_id}.functional_check right-hand side must be a finite numeric "
                    "literal, not an arithmetic expression. Compute the expression in the "
                    "validation script, emit the observed difference, and compare it with 0.")
            threshold_operator, raw_threshold_value = threshold_matches[0]
            if not measurement_name:
                quantities = [gate.group(1) for gate in re.finditer(
                    r"\b([A-Za-z_]\w*)\s*" + _THRESHOLD_RE.pattern, threshold_source)
                    if gate.group(2, 3) == threshold_matches[0]
                    and not threshold_source[:gate.start()].rstrip().endswith(
                        ("+", "-", "*", "/", "%", "."))]
                if len(set(quantities)) != 1:
                    raise ComponentPlanFormatError(
                        f"{component_id}.functional_check must identify one measured quantity")
                measurement_name = quantities[0]
            threshold_value = float(raw_threshold_value)
            if not math.isfinite(threshold_value):
                raise ComponentPlanFormatError(f"{component_id}.functional_check requires a finite threshold")
        depends_raw = raw.get("depends_on") or []
        if not isinstance(depends_raw, list) or not all(
            isinstance(item, str) and item.strip() for item in depends_raw
        ):
            raise ComponentPlanError(f"{component_id}.depends_on must be an array of component IDs")
        if require_design_fields and "acceptance_requires" not in raw:
            raise ComponentPlanError(f"{component_id}.acceptance_requires is required")
        implementation_check = str(raw.get("implementation_check") or "").strip()
        optimization_space = str(raw.get("optimization_space") or "").strip()
        downstream_contract = str(raw.get("downstream_contract") or "").strip()
        if require_design_fields:
            for field_name, value in (
                ("implementation_check", implementation_check),
                ("optimization_space", optimization_space),
                ("downstream_contract", downstream_contract),
            ):
                if not value:
                    raise ComponentPlanError(f"{component_id}.{field_name} is required")
        nodes.append(
            ComponentNode(
                component_id=component_id,
                name=name,
                role=role,
                depends_on=tuple(_component_id(item, 0) for item in depends_raw),
                purpose=purpose,
                inputs=tuple(str(item).strip() for item in inputs_raw),
                outputs=tuple(str(item).strip() for item in outputs_raw),
                acceptance_requires=tuple(
                    _component_id(item, 0) for item in acceptance_requires_raw),
                mechanism=mechanism,
                expected_effect=expected_effect,
                acceptance_check=acceptance_check,
                implementation_check=(implementation_check or
                                      "MIR confirms the implementation matches this component"),
                optimization_space=optimization_space or "no declared optimization space",
                downstream_contract=(downstream_contract or
                                     "provide the declared outputs to dependent components"),
                threshold_operator=threshold_operator,
                threshold_value=threshold_value,
                measurement_name=measurement_name,
            )
        )

    by_id = {node.component_id: node for node in nodes}
    for node in nodes:
        if not node.depends_on:
            _root_checkpoint_token(node)
        missing = [dependency for dependency in node.depends_on if dependency not in by_id]
        if missing:
            raise ComponentPlanError(
                f"{node.component_id} has unknown dependencies: {missing}; "
                f"declared component IDs: {list(by_id)}. Return the complete component graph, "
                "including retained prerequisite contracts; a reference alone does not declare a component.")
        if node.component_id in node.depends_on:
            raise ComponentPlanError(f"{node.component_id} cannot depend on itself")
        unknown_acceptance = [item for item in node.acceptance_requires if item not in by_id]
        if unknown_acceptance:
            raise ComponentPlanError(
                f"{node.component_id} has unknown acceptance prerequisites: {unknown_acceptance}; "
                f"acceptance_requires must contain component IDs from this plan: {sorted(by_id)}. "
                "Artifact/output bindings belong in inputs, not acceptance_requires.")

    indegree = {node.component_id: len(node.depends_on) for node in nodes}
    children: dict[str, list[str]] = {node.component_id: [] for node in nodes}
    for node in nodes:
        for dependency in node.depends_on:
            children[dependency].append(node.component_id)
    queue = [node.component_id for node in nodes if indegree[node.component_id] == 0]
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise ComponentPlanError("component dependencies contain a cycle")

    plan = ComponentPlan(
        method_fingerprint=method_fingerprint(method),
        components=tuple(nodes),
        topological_order=tuple(order),
    )
    if require_design_fields:
        # A direct edge to an ancestor already reached through another parent is only a redundant
        # representation of the same DAG. Canonicalize that edge away so a scientifically sound
        # plan does not spend another LLM round rewriting an equivalent dependency graph. Cycles,
        # missing dependencies and unrelated acceptance prerequisites were rejected above.
        reduced_nodes: list[ComponentNode] = []
        for node in nodes:
            redundant_dependencies = [
                dependency
                for dependency in node.depends_on
                if any(
                    dependency in plan.dependency_closure(other)
                    for other in node.depends_on
                    if other != dependency
                )
            ]
            if redundant_dependencies:
                node = replace(
                    node,
                    depends_on=tuple(
                        dependency for dependency in node.depends_on
                        if dependency not in redundant_dependencies
                    ),
                )
            reduced_nodes.append(node)
        nodes = reduced_nodes
        plan = ComponentPlan(
            method_fingerprint=plan.method_fingerprint,
            components=tuple(nodes),
            topological_order=plan.topological_order,
        )
        for node in nodes:
            closure = set(plan.dependency_closure(node.component_id))
            declared = set(node.acceptance_requires)
            if not closure.issubset(declared):
                raise ComponentPlanError(
                    f"{node.component_id}.acceptance_requires must include its full dependency closure; "
                    f"required IDs: {sorted(closure)}; missing IDs: {sorted(closure - declared)}")
            if not declared.issubset(closure):
                raise ComponentPlanError(
                    f"{node.component_id}.acceptance_requires contains an unrelated component")
            # Free-form text can name a sibling to hold it fixed or exclude it. Whether
            # a test actually activates an undeclared component is a semantic Critic check,
            # not something a substring match can establish.
        covered: set[str] = set()
        for terminal in plan.terminal_components:
            covered.add(terminal)
            covered.update(plan.dependency_closure(terminal))
        if covered != set(plan.topological_order):
            raise ComponentPlanError("terminal components do not cover the full component plan")
    return plan


@dataclass
class ComponentEntry:
    status: str = "pending"
    attempts: int = 0
    infrastructure_attempts: int = 0
    qualified_attempts: int = 0
    parent_checkpoint: str = ""
    parent_checkpoints: list[str] = field(default_factory=list)
    checkpoint: str = ""
    debug_checkpoint: str = ""  # Unqualified worktree; never a dependency or accepted result.
    contract_result: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    last_outcome: str = ""
    metric_value: float | None = None
    quality_value: float | None = None
    quality_lower_is_better: bool = True
    diff: str = ""
    qualified_results: list[dict[str, Any]] = field(default_factory=list)
    search_decision: dict[str, Any] = field(default_factory=dict)
    last_execution_id: str | None = None
    measurement_evidence: dict[str, Any] = field(default_factory=dict)


class ComponentLedger:
    """Atomic JSON ledger; re-opening the same file resumes the active node."""

    def __init__(self, path: str, plan: ComponentPlan, *, root_checkpoint: str = "") -> None:
        self.path = path
        self.plan = plan
        self.root_checkpoint = root_checkpoint
        self._ensure_plan()

    def _initial_data(self) -> dict[str, Any]:
        return {
            "version": 1,
            "plan": self.plan.to_dict(),
            "root_checkpoint": self.root_checkpoint,
            "integration_attempts": 0,
            "entries": {
                component_id: asdict(ComponentEntry())
                for component_id in self.plan.topological_order
            },
        }

    def _load(self) -> dict[str, Any]:
        return self._read_file(self.path)

    @staticmethod
    def _read_file(path: str) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise ComponentPlanError(f"cannot read existing ledger {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("plan"), dict):
            raise ComponentPlanError("existing ledger lacks its recorded plan; original preserved")
        entries = data.get("entries")
        recorded_order = data["plan"].get("topological_order")
        if (not isinstance(entries, dict) or not isinstance(recorded_order, list)
                or set(entries) != set(recorded_order)
                or not recorded_order or not data["plan"].get("method_fingerprint")):
            raise ComponentPlanError("existing ledger has incomplete component entries; original preserved")
        counts = [data.get("integration_attempts")]
        for entry in entries.values():
            if (not isinstance(entry, dict)
                    or entry.get("status") not in {"pending", "active", "accepted", "exhausted"}
                    or (entry.get("status") == "accepted" and not entry.get("checkpoint"))):
                raise ComponentPlanError("existing ledger has invalid component state; original preserved")
            counts.extend(entry.get(key) for key in (
                "attempts", "infrastructure_attempts", "qualified_attempts"))
        if any(type(count) is not int or count < 0 for count in counts):
            raise ComponentPlanError("existing ledger has invalid attempt counts; original preserved")
        return data

    def _save(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)

    def _ensure_plan(self) -> None:
        data = self._load()
        existing = (data.get("plan") or {}).get("method_fingerprint")
        if existing and existing != self.plan.method_fingerprint:
            raise ComponentPlanError("ledger belongs to a different confirmed Method")
        if data and set(data["entries"]) != set(self.plan.topological_order):
            raise ComponentPlanError("ledger entries differ from the confirmed component plan")
        if data and str(data.get("root_checkpoint") or "") != self.root_checkpoint:
            raise ComponentPlanError("ledger belongs to a different root checkpoint")
        if not data:
            self._save(self._initial_data())

    def snapshot(self) -> dict[str, Any]:
        return self._load()

    def record_integration_attempt(self, *, best_result: dict | None = None, execution_id: str | None = None) -> int:
        data = self._load()
        if execution_id and data.get("last_integration_execution_id") == execution_id:
            return int(data.get("integration_attempts") or 0)
        data["last_integration_execution_id"] = execution_id
        data["last_integration_became_best"] = best_result is not None
        attempt = int(data.get("integration_attempts") or 0) + 1
        data["integration_attempts"] = attempt
        if best_result is not None:
            data["integration_best"] = best_result
        self._save(data)
        return attempt

    def entry(self, component_id: str) -> ComponentEntry:
        raw = (self._load().get("entries") or {}).get(component_id)
        if not isinstance(raw, dict):
            raise KeyError(component_id)
        allowed = ComponentEntry.__dataclass_fields__
        return ComponentEntry(**{key: value for key, value in raw.items() if key in allowed})

    def next_ready(self) -> ComponentNode | None:
        data = self._load()
        entries = data.get("entries") or {}
        for component_id in self.plan.topological_order:
            if (entries.get(component_id) or {}).get("status") == "active":
                return self.plan.node(component_id)
        for component_id in self.plan.topological_order:
            entry = entries.get(component_id) or {}
            if entry.get("status") != "pending":
                continue
            node = self.plan.node(component_id)
            if all((entries.get(dep) or {}).get("status") == "accepted" for dep in node.depends_on):
                return node
        return None

    def parent_checkpoints_for(self, component_id: str) -> list[str]:
        node = self.plan.node(component_id)
        if not node.depends_on:
            return [self.root_checkpoint] if self.root_checkpoint else []
        data = self._load()
        entries = data.get("entries") or {}
        checkpoints = [str((entries.get(dep) or {}).get("checkpoint") or "") for dep in node.depends_on]
        if not all(checkpoints):
            raise ComponentPlanError(f"dependencies of {component_id} are not checkpointed")
        return checkpoints

    def parent_checkpoint_for(self, component_id: str) -> str:
        """Human-readable parent identity; execution uses all parent_checkpoints_for()."""
        node = self.plan.node(component_id)
        checkpoints = self.parent_checkpoints_for(component_id)
        if not checkpoints:
            return "baseline"
        if len(checkpoints) == 1:
            return checkpoints[0]
        return "dependency-set:" + ",".join(node.depends_on)

    def begin(self, component_id: str) -> None:
        data = self._load()
        entries = data.get("entries") or {}
        node = self.plan.node(component_id)
        if any((entries.get(dep) or {}).get("status") != "accepted" for dep in node.depends_on):
            raise ComponentPlanError(f"dependencies of {component_id} are not accepted")
        for other_id, other in entries.items():
            if other_id != component_id and other.get("status") == "active":
                raise ComponentPlanError(f"component {other_id} is already active")
        entry = entries[component_id]
        if entry.get("status") == "accepted":
            return
        entry["status"] = "active"
        entry["parent_checkpoints"] = self.parent_checkpoints_for(component_id)
        entry["parent_checkpoint"] = self.parent_checkpoint_for(component_id)
        self._save(data)

    def record_attempt(
        self,
        component_id: str,
        *,
        execution_succeeded: bool,
        contract_passed: bool = False,
        checkpoint: str = "",
        contract_result: dict[str, Any] | None = None,
        error: str = "",
        metric_value: float | None = None,
        metric_lower_is_better: bool = True,
        quality_value: float | None = None,
        quality_lower_is_better: bool = True,
        diff: str = "",
        max_attempts: int = 6,
        failure_outcome: str = "",
        execution_started: bool = True,
        minimum_metric_improvement: float = 0,
        execution_id: str | None = None,
        measurement_evidence: dict | None = None,
    ) -> ComponentEntry:
        data = self._load()
        entry = (data.get("entries") or {}).get(component_id)
        if isinstance(entry, dict) and execution_id and entry.get("last_execution_id") == execution_id:
            return self.entry(component_id)
        if not isinstance(entry, dict) or entry.get("status") != "active":
            raise ComponentPlanError(f"component {component_id} is not active")
        entry["last_execution_id"] = execution_id
        attempt_limit = max(1, int(max_attempts))
        if not execution_succeeded and checkpoint:
            entry["debug_checkpoint"] = checkpoint
        if not execution_started:
            entry["infrastructure_attempts"] = int(entry.get("infrastructure_attempts") or 0) + 1
            entry["last_error"] = str(error or "")
            entry["last_outcome"] = (
                "infrastructure_exhausted" if entry["infrastructure_attempts"] >= attempt_limit
                else "execution_failed")
            # Failed optional optimization does not invalidate an already qualified
            # checkpoint. Without one, exhaustion still requires recovery/reporting.
            if entry["infrastructure_attempts"] >= attempt_limit and entry.get("checkpoint"):
                entry["status"] = "accepted"
            self._save(data)
            return self.entry(component_id)
        entry["attempts"] = int(entry.get("attempts") or 0) + 1
        entry["search_decision"] = {}
        if execution_succeeded and contract_passed:
            if not checkpoint:
                raise ComponentPlanError("a qualified component requires a checkpoint")
            qualified = {
                "checkpoint": checkpoint, "contract_result": contract_result or {},
                "metric_value": metric_value, "quality_value": quality_value,
                "quality_lower_is_better": bool(quality_lower_is_better),
                "diff": str(diff or ""),
            }
            entry.setdefault("qualified_results", []).append(qualified)
            prior_quality = entry.get("quality_value")
            candidate_quality = quality_value
            has_prior = bool(entry.get("checkpoint"))
            better = not has_prior
            if has_prior and candidate_quality is not None and prior_quality is not None:
                # Contract admission already applied the measurement tolerance.
                # Among qualified direct-gain candidates retain the best measured metric.
                tolerance = (minimum_metric_improvement
                             if self.plan.node(component_id).role == "direct_gain" else 0)
                better = (
                    float(prior_quality) - float(candidate_quality) > tolerance
                    if quality_lower_is_better
                    else float(candidate_quality) - float(prior_quality) > tolerance
                )
                if float(candidate_quality) == float(prior_quality):
                    prior_metric = entry.get("metric_value")
                    if metric_value is not None and prior_metric is not None:
                        better = (
                            float(prior_metric) - float(metric_value) > minimum_metric_improvement
                            if metric_lower_is_better
                            else float(metric_value) - float(prior_metric) > minimum_metric_improvement
                        )
            if better:
                entry["checkpoint"] = checkpoint
                entry["contract_result"] = contract_result or {}
                entry["metric_value"] = metric_value
                entry["quality_value"] = candidate_quality
                entry["quality_lower_is_better"] = bool(quality_lower_is_better)
                entry["diff"] = str(diff or "")
                entry["measurement_evidence"] = measurement_evidence or {}
            entry["qualified_attempts"] = int(entry.get("qualified_attempts") or 0) + 1
            entry["last_error"] = ""
            entry["last_outcome"] = "qualified" if not has_prior else (
                "improved" if better else "not_improved")
            # A non-improving follow-up is a default stopping heuristic, not proof of
            # optimality. Honor the existing explicit search goal within the same budget.
            from .execution_policy import efficient_execution
            stop_on_non_improvement = os.environ.get("AUTORESEARCH_FORCE_KEEP_SEARCHING") != "1"
            entry["status"] = (
                "accepted"
                if entry["attempts"] >= attempt_limit
                or (not efficient_execution() and has_prior and not better
                    and stop_on_non_improvement)
                else "active"
            )
        else:
            entry["last_error"] = str(error or "")
            entry["last_outcome"] = str(failure_outcome or (
                "execution_failed" if not execution_succeeded else "contract_failed"))
            if entry["attempts"] >= attempt_limit:
                # A previously qualified checkpoint is still valid even if the last attempted
                # improvement failed.  Without one, the component must return to method revision.
                entry["status"] = "accepted" if entry.get("checkpoint") else "exhausted"
            else:
                entry["status"] = "active"
        self._save(data)
        return self.entry(component_id)

    def decide_qualified(self, component_id: str, *, result_ref: str, action: str,
                         reason: str, next_experiment: str = "",
                         selected_checkpoint: str = "best") -> ComponentEntry:
        """Advance or keep improving a verified qualified node, retaining its selected checkpoint."""
        data = self._load()
        entry = (data.get("entries") or {}).get(component_id)
        if not isinstance(entry, dict) or entry.get("status") not in {"active", "accepted"} or not entry.get("checkpoint"):
            raise ComponentPlanError(f"component {component_id} has no active qualified checkpoint")
        if action not in {"advance", "improve"} or (action == "improve" and not next_experiment):
            raise ComponentPlanError("invalid qualified-component decision")
        if entry["status"] == "accepted" and (
                action != "advance" or self.plan.node(component_id).role != "indirect_gain"):
            raise ComponentPlanError("accepted component can only finalize an indirect checkpoint")
        if entry["status"] == "accepted" and entry.get("search_decision"):
            raise ComponentPlanError("accepted component checkpoint is already finalized")
        if entry["status"] == "accepted" and (
                data.get("integration_attempts") or any(
                    component_id in self.plan.node(child_id).depends_on
                    and child.get("status") != "pending"
                    for child_id, child in (data.get("entries") or {}).items())):
            raise ComponentPlanError("cannot change a parent checkpoint after downstream work starts")
        if selected_checkpoint != "best":
            if self.plan.node(component_id).role != "indirect_gain":
                raise ComponentPlanError("direct gain must retain the host-best checkpoint")
            selected = next((item for item in entry.get("qualified_results") or []
                             if item.get("checkpoint") == selected_checkpoint), None)
            if selected is None:
                raise ComponentPlanError("selected checkpoint is not a qualified component result")
            entry.update({key: selected[key] for key in (
                "checkpoint", "contract_result", "metric_value", "quality_value",
                "quality_lower_is_better", "diff")})
        entry["search_decision"] = {
            "result_ref": result_ref, "action": action, "reason": reason[:1500],
            "next_experiment": next_experiment[:1500],
            "selected_checkpoint": entry["checkpoint"],
        }
        if action == "advance":
            entry["status"] = "accepted"
        self._save(data)
        return self.entry(component_id)

    def all_accepted(self) -> bool:
        entries = self._load().get("entries") or {}
        return (set(entries) == set(self.plan.topological_order)
                and all(entries[item].get("status") == "accepted"
                        for item in self.plan.topological_order))


def evaluate_component_contract(
    node: ComponentNode,
    *,
    parent_metric: float | None,
    candidate_metric: float | None,
    lower_is_better: bool,
    ablation_report: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Host-side component gate; never trusts a sandbox self-reported verdict."""

    if node.role == "direct_gain":
        for item in ablation_report or []:
            if not isinstance(item, dict) or item.get("component") not in {node.component_id, node.name}:
                continue
            evidence = item.get("parent_evidence") or {}
            if (isinstance(evidence, dict) and evidence.get("host_bound") is True
                    and evidence.get("failed_gates")
                    and evidence.get("reproduced_ineligible_baseline") is not True):
                return {"passed": False, "parent_evidence": evidence,
                        "reason": "parent/control explicitly failed correctness; its scalar is not a valid "
                                  "performance control. This is not an absence of candidate execution. "
                                  "Reconcile the control/acceptance design or measure the required valid parent; "
                                  "do not silently substitute the frozen baseline. Preserve both measurements."}
        try:
            measurement, operator, threshold = _direct_gain_predicate(node.acceptance_check)
        except ComponentPlanFormatError as exc:
            return {"passed": False, "plan_invalid": True, "reason": str(exc)}
        parent_evidence = None
        matches = [
            item for item in (ablation_report or [])
            if isinstance(item, dict)
            and str(item.get("component") or "").strip() in {
                node.component_id, node.name}
        ]
        if len(matches) == 1:
            # Keep an explicit control reference distinct from the candidate
            # run. Older reports used one evidence field for the control.
            evidence = matches[0].get("parent_evidence", matches[0].get("evidence"))
            if (isinstance(evidence, dict)
                    and evidence.get("host_bound") is True
                    and str(evidence.get("source") or "").strip()
                    and str(evidence.get("ref") or "").strip()):
                parent_metric = matches[0].get("parent_metric")
                parent_evidence = evidence
        try:
            if isinstance(parent_metric, bool) or isinstance(candidate_metric, bool):
                raise ValueError("boolean is not a metric")
            parent = float(parent_metric)
            candidate = float(candidate_metric)
            if not math.isfinite(parent) or not math.isfinite(candidate):
                raise ValueError("non-finite metric")
        except (TypeError, ValueError):
            return {"passed": False, "reason": "direct-gain component lacks finite parent/candidate metrics"}
        improvement = parent - candidate if lower_is_better else candidate - parent
        actual = improvement if measurement == "signed_metric_improvement" else candidate
        meets_threshold = {
            "<": actual < threshold, "<=": actual <= threshold, "==": actual == threshold,
            ">=": actual >= threshold, ">": actual > threshold,
        }[operator]
        passed = improvement > 0
        return {
            "passed": passed,
            "measurement": "signed_metric_improvement",
            "actual": improvement,
            "expected": {"operator": ">", "value": 0.0},
            "acceptance_check_result": {
                "measurement": measurement, "actual": actual,
                "expected": {"operator": operator, "value": threshold},
                "passed": meets_threshold,
            },
            "parent_metric": parent,
            "candidate_metric": candidate,
            **({"parent_evidence": parent_evidence} if parent_evidence else {}),
            "reason": "official metric improved and the frozen acceptance threshold passed"
            if passed and meets_threshold else
            "official metric improved, but the frozen CAP expected gain was not reached; "
            "retain this shortfall for Method reflection"
            if passed else "direct-gain component did not improve over its parent checkpoint",
        }

    matches = [
        item for item in (ablation_report or [])
        if isinstance(item, dict)
        and str(item.get("component") or "").strip() in {node.component_id, node.name}
    ]
    if len(matches) != 1:
        return {"passed": False, "reason": "indirect-gain component needs exactly one measurement"}
    item = matches[0]
    if re.search(r"\b(signed_metric_improvement|candidate_metric|metric delta)\s*"
                 + _THRESHOLD_RE.pattern, node.acceptance_check):
        return {"passed": False, "plan_invalid": True,
                "reason": "task-performance quantity requires a direct_gain parent/candidate comparison"}
    evidence = item.get("evidence")
    try:
        actual = float(item.get("actual"))
        evidence_ok = (
            isinstance(evidence, dict)
            and item.get("_host_bound") is True
            and not isinstance(item.get("actual"), bool) and math.isfinite(actual)
            and bool(str(evidence.get("source") or "").strip())
            and bool(str(evidence.get("ref") or "").strip())
        )
    except (TypeError, ValueError):
        actual, evidence_ok = 0.0, False
    if not evidence_ok or node.threshold_value is None:
        return {"passed": False, "reason": "indirect-gain measurement lacks grounded evidence"}
    path = str(evidence.get("json_pointer") or "")
    quantity = path.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    if not path.startswith("/") or quantity != node.measurement_name:
        return {"passed": False,
                "reason": f"measurement quantity mismatch: bind {node.measurement_name!r} "
                          "to its named field in saved runtime output; unrelated scalars "
                          "and unspecified aliases cannot satisfy this contract"}
    expected = float(node.threshold_value)
    # A declared different threshold can identify an obsolete check after plan revision.
    # Do not reinterpret that observation under the new plan as a failed hypothesis.
    # Omission remains supported: the host, not the sandbox, owns the threshold.
    reported = item.get("expected")
    if reported is not None:
        try:
            matches_plan = (
                isinstance(reported, dict)
                and reported.get("operator") == node.threshold_operator
                and float(reported.get("value")) == expected
            )
        except (TypeError, ValueError):
            matches_plan = False
        if not matches_plan:
            return {
                "passed": False,
                "reason": (
                    "reported acceptance threshold does not match the active ComponentPlan; "
                    "rerun the current frozen check, not a previous plan's diagnostic: "
                    + node.acceptance_check
                ),
            }
    passed = {
        "<": actual < expected,
        "<=": actual <= expected,
        ">": actual > expected,
        ">=": actual >= expected,
        "==": actual == expected,
    }[node.threshold_operator]
    return {
        "passed": passed,
        "measurement": "indirect_gain_check",
        "measurement_name": node.measurement_name,
        "actual": actual,
        "expected": {"operator": node.threshold_operator, "value": expected},
        "evidence": evidence,
        "reason": "indirect-gain threshold passed" if passed else "indirect-gain threshold failed",
    }


_CHECKPOINT_TOKEN_RE = re.compile(r"\bcheckpoint-[0-9a-f]{6,64}\b", re.IGNORECASE)


def _root_checkpoint_token(node: ComponentNode) -> str:
    tokens = set(_CHECKPOINT_TOKEN_RE.findall(" ".join(node.inputs)))
    if len(tokens) > 1:
        raise ComponentPlanError(
            f"{node.component_id} names multiple root checkpoints; inputs may name at most "
            "one executable parent checkpoint. Put additional code-copy references in "
            "mechanism or reference_sources; depends_on and merged_from are component IDs.")
    return next(iter(tokens)).lower() if tokens else ""


def explicit_root_checkpoint(plan: ComponentPlan, history: list[dict[str, Any]], *,
                             baseline_checkpoint: str = "",
                             reconciled_checkpoint: str = "") -> str:
    """Bind a named root to verified history or a host-validated frozen source."""
    roots = [node for node in plan.components if not node.depends_on]
    requested = []
    for node in roots:
        token = _root_checkpoint_token(node)
        if not token:
            requested.append("")
            continue
        matches = {
            str(item.get("selected_checkpoint") or item.get("checkpoint"))
            for item in history if isinstance(item, dict)
            and item.get("contract_passed") is True
            and item.get("mir_verdict") == "faithful"
            and item.get("verification_verdict") == "verified"
            and (item.get("acceptance_result") or {}).get("passed") is True
            and os.path.basename(str(item.get("selected_checkpoint") or item.get("checkpoint") or ""))
            .lower().startswith(token)
        }
        if baseline_checkpoint and os.path.basename(baseline_checkpoint).lower().startswith(token):
            matches.add(baseline_checkpoint)
        if (reconciled_checkpoint
                and os.path.basename(reconciled_checkpoint).lower().startswith(token)):
            matches.add(reconciled_checkpoint)
        if len(matches) != 1:
            raise ComponentPlanError(
                f"{node.component_id} root checkpoint {token} has no unique verified, "
                "contract-passing component observation. inputs declares executable dependencies: "
                "every checkpoint token there is resolved, including tokens in negated prose. "
                "Remove excluded checkpoint references from inputs and keep their exclusion "
                "or exposure notes in mechanism or reference_sources. If no prior checkpoint "
                "is required, declare only the fresh repository or frozen baseline inputs; "
                "do not qualify or inherit the excluded checkpoint.")
        requested.append(next(iter(matches)))
    if any(requested) and len(set(requested)) != 1:
        raise ComponentPlanError("root components require incompatible starting checkpoints")
    return requested[0] if requested else ""


def ledger_for_workroot(workroot: str, plan: ComponentPlan, *, execution_identity: str | None = None,
                        root_checkpoint: str = "") -> ComponentLedger:
    root = workroot or "."
    legacy_path = os.path.join(root, "component_ledger.json")
    legacy = ComponentLedger._read_file(legacy_path)
    if ((legacy.get("plan") or {}).get("method_fingerprint") == plan.method_fingerprint
            and (execution_identity is None or legacy.get("execution_identity") == execution_identity)
            and str(legacy.get("root_checkpoint") or "") == root_checkpoint):
        return ComponentLedger(legacy_path, plan, root_checkpoint=root_checkpoint)

    # A formal component revision creates a new Method fingerprint.  Give that revision its own
    # crash-safe ledger instead of either inheriting incompatible checkpoints or overwriting the
    # prior component evidence.  Legacy in-flight sessions still resume from component_ledger.json.
    suffix = plan.method_fingerprint.removeprefix("sha256:")[:16]
    if execution_identity:
        suffix += "-" + hashlib.sha256(execution_identity.encode()).hexdigest()[:16]
    if root_checkpoint:
        suffix += "-" + hashlib.sha256(root_checkpoint.encode()).hexdigest()[:12]
    ledger = ComponentLedger(os.path.join(root, f"component_ledger-{suffix}.json"), plan,
                             root_checkpoint=root_checkpoint)

    # A ComponentPlan-only revision may repair one node's dependency or acceptance contract while
    # leaving earlier, already accepted nodes unchanged.  Re-running those nodes wastes real compute
    # and loses valid evidence.  Inherit only an exact component-contract match with an accepted,
    # checkpointed entry from this same session workroot, including the tested dependency context.
    # A changed component or ancestor never qualifies and starts pending in the revised ledger.
    target = ledger._load()
    if execution_identity is not None:
        target["execution_identity"] = execution_identity
        ledger._save(target)
    inherited: list[dict[str, str]] = []
    current_nodes = {node.component_id: node.to_dict() for node in plan.components}
    candidates = sorted(
        glob.glob(os.path.join(root, "component_ledger*.json")),
        key=lambda path: os.path.getmtime(path), reverse=True,
    )
    for candidate_path in candidates:
        if os.path.abspath(candidate_path) == os.path.abspath(ledger.path):
            continue
        candidate = ComponentLedger._read_file(candidate_path)
        if execution_identity is not None and candidate.get("execution_identity") != execution_identity:
            continue
        if str(candidate.get("root_checkpoint") or "") != root_checkpoint:
            continue
        prior_nodes = {
            str(item.get("component_id") or ""): item
            for item in ((candidate.get("plan") or {}).get("components") or [])
            if isinstance(item, dict)
        }
        prior_entries = candidate.get("entries") or {}
        for component_id in plan.topological_order:
            node_contract = current_nodes[component_id]
            current_entry = (target.get("entries") or {}).get(component_id) or {}
            prior_entry = prior_entries.get(component_id) or {}
            if current_entry.get("status") != "pending":
                continue
            if prior_nodes.get(component_id) != node_contract:
                continue
            if prior_entry.get("status") != "accepted" or not prior_entry.get("checkpoint"):
                continue
            if (plan.node(component_id).role == "indirect_gain"
                    and ((prior_entry.get("contract_result") or {}).get("measurement_review") or {}).get("verdict") != "pass"):
                continue
            try:
                from .checkpoint_artifacts import checkpoint_recoverable
                if not checkpoint_recoverable(prior_entry["checkpoint"]):
                    continue
            except (KeyError, TypeError):
                # Keep the old observation intact; it is not a recoverable new-plan node.
                continue
            closure = plan.dependency_closure(component_id)
            if any(
                prior_nodes.get(dep) != current_nodes[dep]
                or (target["entries"].get(dep) or {}).get("status") != "accepted"
                or (target["entries"].get(dep) or {}).get("checkpoint")
                != (prior_entries.get(dep) or {}).get("checkpoint")
                for dep in closure
            ):
                continue
            parents = [target["entries"][dep]["checkpoint"]
                       for dep in plan.node(component_id).depends_on]
            if not parents and root_checkpoint:
                parents = [root_checkpoint]
            if parents and prior_entry.get("parent_checkpoints") != parents:
                continue
            target["entries"][component_id] = dict(prior_entry)
            inherited.append({
                "component_id": component_id,
                "ledger": os.path.basename(candidate_path),
                "checkpoint": str(prior_entry["checkpoint"]),
            })
        if all((target.get("entries") or {}).get(component_id, {}).get("status") != "pending"
               for component_id in current_nodes):
            break
    if inherited:
        target["inherited_accepted_components"] = inherited
        ledger._save(target)
    return ledger


def component_memory_packet(
    history: list[dict[str, Any]], *, component_id: str,
    dependency_closure: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Keep every attempt visible without copying every large record into each prompt.

    The active component and its ancestors retain their bounded host-derived records.  Every
    unrelated branch still appears once, but only as a routing summary with artifact references.
    """
    detailed_ids = {component_id, *dependency_closure}
    detailed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for raw in history:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        # A finite, contract-passing observation can still have been rejected by an older
        # whole-Method implementation review.  Preserve the measurement and artifact refs, but
        # do not replay that review's command-like remediation into a single-component retry:
        # it may instruct the agent to implement unready siblings.
        if (isinstance(item.get("metric"), (int, float))
                and item.get("contract_passed") is True
                and str(item.get("mir_verdict") or "") not in {"", "faithful"}
                and "method implementation review" in str(
                    item.get("failure_reason") or "").lower()):
            item["failure_reason"] = (
                "A prior implementation review did not accept attribution; the finite metric, "
                "diff, and review verdict remain recorded. Re-evaluate this artifact only "
                "against the currently scheduled component contract.")
        if str(item.get("component_id") or "") in detailed_ids:
            detailed.append(item)
            continue
        summaries.append({
            key: item.get(key)
            for key in (
                "sequence", "component_id", "round", "attempt", "status", "operator",
                "metric", "contract_passed", "mir_verdict", "became_best", "run_ref",
                "checkpoint", "diff_ref", "failure_reason",
            )
            if key in item
        })
    return {
        "current_component": component_id,
        "dependency_closure": list(dependency_closure),
        "current_and_ancestor_rounds": detailed,
        "other_branch_round_summaries": summaries,
        "history_entry_count": len(detailed) + len(summaries),
    }


__all__ = [
    "ComponentEntry",
    "ComponentLedger",
    "ComponentNode",
    "ComponentPlan",
    "ComponentPlanError",
    "build_component_plan",
    "evaluate_component_contract",
    "component_memory_packet",
    "ledger_for_workroot",
    "method_fingerprint",
]
