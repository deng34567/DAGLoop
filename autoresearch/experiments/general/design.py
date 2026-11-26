"""Task 1: turn a USER's research question into a structured, code-runnable
experiment design (the P1 "general honest experiment" data contract).

``design_experiment(llm, question) -> dict`` calls the injected LLM (same pattern
as the other agents: ``llm.chat(... response_format="json_object")`` ->
``json.loads(extract_json(content))``) and validates the returned JSON has the 7
required fields. PURE w.r.t. transport: no EventBus / Store / network here.

The design is deliberately CODE-ORIENTED: instances must be seed-reproducible and
the metric must be computable by independent code, so the downstream harness can
score a method honestly.
"""
from __future__ import annotations

import json
import re
import shlex

from autoresearch.agents._reuse.json_extract import extract_json

# A dependency name is interpolated into a shell `pip install` command, so it must
# be a plain package token. Anything outside this set (spaces, ``;``, ``$(...)``,
# ``/`` ...) is dropped BEFORE it can reach a shell -- see ``sanitize_dependencies``.
_SAFE_PKG = re.compile(r"^[A-Za-z0-9._-]+$")

REQUIRED_FIELDS = (
    "question",         # the user's original request (verbatim-ish)
    "task",             # the formalised task (e.g. 1D bin packing, capacity C, size dist)
    "instance_spec",    # how to generate synthetic instances from a seed
    "baseline",         # the baseline to beat (e.g. First-Fit)
    "method_interface", # signature the method implements, e.g. solve(instance)->solution
    "metric",           # the metric + validity criterion (computable by code)
    "success_criterion",# what counts as "better"
)


class DesignError(Exception):
    """The LLM did not return a well-formed 7-field ExperimentDesign."""


def sanitize_dependencies(deps) -> list:
    """Return the subset of ``deps`` that are SAFE to interpolate into a shell
    ``pip install``: string tokens matching ``^[A-Za-z0-9._-]+$``. A missing /
    non-list ``deps`` yields ``[]``; malicious/garbled entries are dropped."""
    if not isinstance(deps, list):
        return []
    return [d for d in deps if isinstance(d, str) and _SAFE_PKG.match(d)]


def pip_install_command(deps):
    """Build a ``pip install -q ...`` command for the SANITIZED, shell-quoted
    ``deps``, or ``None`` if there is nothing safe to install."""
    safe = sanitize_dependencies(deps)
    if not safe:
        return None
    return "pip install -q " + " ".join(shlex.quote(d) for d in safe)


_SYSTEM_PROMPT = (
    "You are a research-experiment designer. Turn the user's research question "
    "into a CONCRETE experiment that can be run as real code with HONEST, "
    "code-computable verification.\n\n"
    "Respond with a SINGLE JSON object with EXACTLY these fields:\n"
    "  - question: the user's original request.\n"
    "  - task: the formalised task (be specific: e.g. '1D bin packing, "
    "capacity C=1.0, item sizes ~ U(0,0.7)').\n"
    "  - instance_spec: how to generate synthetic instances FROM A SEED so runs "
    "are reproducible (must use the seed; e.g. numpy default_rng(seed)).\n"
    "  - baseline: a concrete, real baseline to beat, naming EXACTLY ONE method "
    "(e.g. 'First-Fit'); NEVER a list of methods and NEVER a survey of heuristics. "
    "Use the baseline the user's question NAMES if it names one. The baseline must be "
    "a SIMPLE reference method the proposed method can plausibly BEAT -- NEVER the "
    "exact optimum / brute-force / oracle / ground-truth solution (e.g. exact DP, an "
    "ILP/MILP solver, exhaustive search): nothing beats the optimum, so an optimal "
    "baseline makes the experiment unwinnable. (You MAY still normalize the metric "
    "against the optimum; just don't make the optimum the baseline.)\n"
    "  - method_interface: the exact contract BOTH the harness and the method must "
    "obey, so two independently-written files agree. Pin it UNAMBIGUOUSLY: (a) the "
    "EXACT structure of one `instance` object (name every field and its type, e.g. "
    "\"instance is a dict {'capacity': float, 'items': list[float]}\"), and (b) the "
    "EXACT structure of the `solution` object solve returns and validate checks "
    "(e.g. \"solution is a list of bins; each bin is a list of the integer INDICES "
    "into items placed in that bin\" -- prefer INDICES over raw values so validate "
    "is unambiguous). Give a tiny concrete example of both. Form: solve(instance) -> "
    "solution.\n"
    "  - metric: the metric AND the validity criterion, both computable by "
    "independent code (e.g. 'number of bins used, lower is better; valid = all "
    "items placed and no bin exceeds capacity').\n"
    "  - success_criterion: what makes the method 'better' than the baseline. Keep the "
    "PROPOSED METHOD and the BASELINE in the SAME roles the question assigns them: state "
    "it as the proposed method achieving a better metric than the baseline -- do NOT swap "
    "which one is the method vs the baseline, and do NOT invert the direction.\n"
    "  - lower_is_better: a JSON boolean -- true if a SMALLER metric value is better "
    "(e.g. fewer bins, lower error, shorter tour, fewer colors), false if a LARGER "
    "value is better (e.g. more value/profit, higher accuracy, higher throughput). "
    "It MUST match the metric's direction above.\n"
    "  - gpu: a JSON boolean -- USE YOUR JUDGEMENT about whether ACTUALLY RUNNING this "
    "experiment would genuinely need or strongly benefit from a GPU. Decide from what the "
    "experiment COMPUTES, not from keywords: true for e.g. training OR running a neural "
    "network, large dense tensor/matrix computation, CUDA-accelerated simulation, "
    "deep-learning inference; false for ordinary CPU algorithms (sorting, bin packing, TSP, "
    "graph, small-scale statistics/numerics). If true, also put the GPU framework you will "
    "use (e.g. \"torch\", \"cupy\") in dependencies.\n"
    "  - io_contract: a JSON object with a CONCRETE tiny EXAMPLE so the "
    "independently-written harness and method agree on the EXACT shape: "
    "{\"instance_example\": <one instance object EXACTLY as make_instances "
    "produces one>, \"solution_example\": <one VALID solution for THAT instance, in "
    "the EXACT shape solve returns and validate expects>}. Prefer integer INDICES "
    "over raw values in the solution. This example is the public SHAPE contract "
    "(structure only, not an answer key).\n"
    "  - dependencies: a JSON list of the THIRD-PARTY pip package names the "
    "experiment will need beyond the Python standard library (e.g. [\"numpy\"], "
    "or [\"networkx\", \"pandas\"], or [] if it is pure stdlib). List ONLY real "
    "pip packages; do NOT list stdlib modules (json, math, random, ...).\n\n"
    "The instances MUST be seed-reproducible and the metric MUST be computable by "
    "independent code. Do not invent data files; generate everything from the seed.\n\n"
    "KEEP IT MINIMAL AND TRACTABLE (critical): design EXACTLY ONE small self-contained "
    "comparison -- a SINGLE instance distribution, a MODEST fixed scale (about n=100, "
    "at most a few hundred; runs in seconds), EXACTLY ONE proposed method vs ONE "
    "baseline, and ONE scalar metric. Do NOT design a multi-distribution, multi-scale, "
    "or multi-heuristic study, and do NOT over-scope even if the question is broad -- "
    "pick the single most representative comparison and scope everything else out, so a "
    "single solve(instance) function and a single harness can implement and run it easily.\n\n"
    "DISCRIMINATING INSTANCES (critical): choose the instance distribution AND scale so "
    "the method and baseline CAN differ MEANINGFULLY. Avoid a trivially-easy regime where "
    "both are essentially optimal or both hit the same floor -- e.g. a function so smooth "
    "that every integration rule reaches machine precision (~1e-16), instances so tiny that "
    "every heuristic ties, or a scale where the metric saturates. Pick a hardness/scale "
    "where a better method has room to show a real, measurable gap over the baseline.\n\n"
    "GPU EXPERIMENTS: whenever gpu=true, keep the work SMALL so a run finishes in seconds, "
    "and make sure the harness holds the GROUND TRUTH the method is scored against (never "
    "put the answer in the instance). ONE COMMON GPU CASE -- a TRAINING experiment (e.g. "
    "'does warmup / this optimizer / this regularizer improve test error vs a baseline'): "
    "design it as a TEACHER-STUDENT task so "
    "the method cannot see the answer key: the harness holds a FIXED teacher network (fixed "
    "seed) as the ground truth; each `instance` is training data (noisy samples of the "
    "teacher) PLUS the test INPUTS, but NOT the teacher or the clean test targets. The "
    "method trains a student on the training data and returns its PREDICTIONS on the test "
    "inputs; the metric is the average error between those predictions and the teacher's "
    "CLEAN outputs (which only the harness computes), lower is better. Keep it SMALL (a "
    "small MLP, a few hundred steps) so a run finishes in seconds. instance_spec/io_contract "
    "must make instance = {train_x, train_y, test_x} and solution = a list of floats."
)


def _user_message(question: str) -> str:
    return (
        f"User research question:\n{question}\n\n"
        "Design the experiment as a single JSON object with the fields "
        "described in the system prompt (including the dependencies list). The "
        "method interface must be solve(instance) -> solution."
    )


# Strong-signal words that a design is a multi-method SURVEY rather than the
# required ONE method vs ONE baseline. Detecting these triggers ONE narrowing
# re-design (bounded + safe: a false positive costs one extra call, never loops).
_MULTI_HINT = re.compile(
    r"\b(multiple|several|various|systematically\s+compar|survey|classical\s+heuristics)\b"
    r"|各种|多种|多个|逐一|系统(性|地)?比较|一系列",
    re.I,
)


# Direction inference fallback (used only when the LLM omits lower_is_better).
_HIGHER_BETTER = re.compile(
    r"higher\s+is\s+better|larger\s+is\s+better|more\s+is\s+better|maximi[sz]|"
    r"greater\s+is\s+better|越大越好|越高越好|越多越好", re.I)


def _infer_lower_is_better(design: dict) -> bool:
    """Infer the metric direction from text. Higher-is-better phrasing -> False;
    otherwise default True (matches the historical fewer-bins/lower-error case)."""
    text = " ".join(str(design.get(k) or "") for k in ("metric", "success_criterion", "task"))
    return not bool(_HIGHER_BETTER.search(text))


def _overscoped_reason(design: dict) -> str | None:
    """Return a short reason if the design reads as a multi-method comparison/survey
    (harder to produce ONE valid solution for), else None. Conservative: only fires
    on strong signals; a false positive is harmless (one extra narrowing call)."""
    text = " ".join(
        str(design.get(k) or "")
        for k in ("task", "baseline", "success_criterion", "method_interface")
    )
    if _MULTI_HINT.search(text):
        return "the design reads as a multi-method comparison/survey"
    # A baseline field that itself names 3+ separated methods is a survey baseline.
    parts = [p for p in re.split(r"[,、;/]|\bvs\.?\b|\band\b|\b与\b|\b和\b", design.get("baseline") or "")
             if p.strip()]
    if len(parts) >= 3:
        return f"the 'baseline' field names {len(parts)} methods (a baseline must be exactly one)"
    return None


# A baseline that is the EXACT OPTIMUM / an oracle is UNBEATABLE, so the proposed
# method can never "beat" it -> the run is meaningless (always fail). The design
# LLM does this when a problem has a natural optimum (e.g. knapsack -> exact DP):
# it silently swaps the user's simple baseline for the optimum. Detect it in the
# BASELINE field ONLY (the same word in the METRIC, e.g. "ratio vs optimal", is a
# fine normalization) and issue ONE corrective re-design.
_UNBEATABLE_BASELINE = re.compile(
    r"\boptimal\b|\bexact\b|brute[\s-]?force|\boracle\b|ground[\s-]?truth|exhaustive"
    r"|\bILP\b|\bMILP\b|branch[\s-]and[\s-]bound|最优|精确解|穷举|暴力",
    re.I,
)


def _unbeatable_baseline_reason(design: dict) -> str | None:
    """Return a short reason if the BASELINE is an exact-optimum / oracle solver (which
    the proposed method can never beat -> the experiment is unwinnable), else None.
    Only the ``baseline`` field is inspected: 'optimal' in the metric (a normalization
    reference) is fine. Conservative: a false positive costs one extra design call."""
    base = str(design.get("baseline") or "")
    if _UNBEATABLE_BASELINE.search(base):
        return f"the 'baseline' names an exact-optimum/oracle solver ({base.strip()[:80]!r})"
    return None


def _fix_baseline_message(question: str, reason: str) -> str:
    return (
        f"User research question:\n{question}\n\n"
        f"Your previous design used an UNBEATABLE baseline: {reason}. The proposed method "
        "cannot beat the exact optimum / an oracle, so that comparison is meaningless "
        "(it can only ever fail). REDESIGN with a SIMPLE, REAL reference method as the "
        "'baseline' -- the one the user's question names if it names one (e.g. a naive/"
        "greedy/first-fit heuristic), NOT the optimal/exact/brute-force/oracle solution. "
        "Keep the PROPOSED METHOD and the BASELINE exactly as the question assigns them "
        "(the method should be able to BEAT the baseline). You MAY still measure the "
        "metric relative to the optimum, but the baseline itself must be beatable. "
        "Return the single JSON object with the required fields."
    )


# A DETERMINISTIC role-inversion guard (code-enforced, NOT fail-open like the LLM critic
# below): a success_criterion that frames the PROPOSED METHOD as suboptimal / worse / inferior
# means the method is set up to LOSE to the baseline -- the exact degenerate design the LLM
# critic sometimes lets through (observed live: "...establishing that EST is suboptimal").
# Conservative regex (unambiguous "method is/shown/... suboptimal|worse|inferior" or
# "establishing ... suboptimal|inferior" only), so a false positive is rare and costs just one
# extra bounded re-design.
_ROLE_INVERSION = re.compile(
    r"(method|proposed|candidate|approach)[^.。]{0,70}"
    r"(is|are|be|being|remain\w*|achiev\w*|prov\w*|shown|demonstrat\w*)[^.。]{0,30}"
    r"(suboptimal|inferior|worse|underperform\w*)"
    r"|establish\w*[^.。]{0,40}(suboptimal|inferior|worse)"
    r"|(方法|所提|候选)[^。.]{0,30}(次优|更差|劣于|不如)",
    re.I,
)


def _method_set_to_lose_reason(design: dict) -> str | None:
    """Deterministic role-inversion guard: return a reason if the success_criterion frames the
    PROPOSED METHOD as suboptimal / worse / inferior to the baseline (the method is set up to
    LOSE). Only the success_criterion is inspected. Conservative -- see the regex note above."""
    if _ROLE_INVERSION.search(str(design.get("success_criterion") or "")):
        return ("the success_criterion frames the PROPOSED METHOD as suboptimal / worse than "
                "the baseline (role inversion -- the method is set up to lose)")
    return None


def _fix_role_inversion_message(question: str, reason: str) -> str:
    return (
        f"User research question:\n{question}\n\n"
        f"Your previous design is ROLE-INVERTED: {reason}. An experiment must give the PROPOSED "
        "METHOD a genuine chance to WIN: write the success_criterion so the METHOD achieves a "
        "BETTER metric value than the baseline (in the metric's own better-direction), NOT that "
        "it is worse/suboptimal/inferior. Put the algorithm the user actually wants to test as "
        "the PROPOSED METHOD and a SIMPLE beatable reference as the baseline, and phrase the "
        "success_criterion so the method BEATS the baseline. Return the single JSON object with "
        "the required fields."
    )


# T6: an ML/statistical experiment (metric = accuracy/error/loss/F1/... where overfitting is
# possible) MUST measure its metric on a HELD-OUT split -- a train-only metric overfits and the
# evaluator rightly flags "missing held_out validation". Fires ONLY for ML-like metrics AND
# only when no held-out/validation is mentioned, so combinatorial designs (bin count, tour
# length, #intervals) are never touched.
_ML_METRIC = re.compile(
    r"accuracy|error\s*rate|\berror\b|\bloss\b|\bf1\b|\bauc\b|precision|recall|\brmse\b|\bmae\b"
    r"|perplexity|\bbpb\b|silhouette|\br2\b|r\^2|准确率|误差|损失",
    re.I,
)
_HELDOUT = re.compile(
    r"held.?out|hold.?out|validation|test\s*set|test\s*split|train[\s/_-]?test|cross.?valid"
    r"|k.?fold|generaliz|out.?of.?sample|留出|验证集|测试集|交叉验证|泛化",
    re.I,
)


def _missing_heldout_reason(design: dict) -> str | None:
    """Return a reason if the design measures an ML/statistical metric but specifies NO held-out
    / validation split (train-only metrics overfit). Returns None for non-ML metrics
    (combinatorial counts/lengths, where held-out is not applicable) or when a split is named."""
    if not _ML_METRIC.search(str(design.get("metric") or "")):
        return None
    text = " ".join(str(design.get(k) or "")
                    for k in ("metric", "success_criterion", "instance_spec", "task"))
    if _HELDOUT.search(text):
        return None
    return ("the design measures an ML metric but specifies no held-out / validation split "
            "(a metric measured on the training data overfits)")


def _fix_heldout_message(question: str, reason: str) -> str:
    return (
        f"User research question:\n{question}\n\n"
        f"Your previous design is missing a validation protocol: {reason}. REDESIGN so the "
        "instance_spec and success_criterion explicitly use a HELD-OUT split: fit/train the "
        "method AND all preprocessing (scaling, feature selection, imputation) ONLY on the "
        "training partition, then measure the metric on a SEPARATE held-out TEST set (or k-fold "
        "cross-validation) -- never on the training data. Add a positive and a negative control "
        "where sensible. Return the single JSON object with the required fields."
    )


# T7 (Data-Source Fidelity): when the user's question explicitly requires an external or real
# data source, the design must actually load it rather than silently substituting synthetic data.
# Detect the requirement from generic source language or a supplied data URL/path; never maintain
# a list of project, dataset or benchmark names here.
_EXPLICIT_DATA_SOURCE = re.compile(
    r"\b(?:real[- ]world|real|public|official|external)\s+(?:data|dataset|corpus)\b"
    r"|\b(?:dataset|data\s+set|corpus|data\s+source|benchmark\s+data)\b"
    r"|(?:真实|公开|官方|外部)(?:数据|数据集|语料)"
    r"|(?:数据集|数据源|语料库|基准数据)",
    re.I,
)
_SUPPLIED_DATA_LOCATION = re.compile(
    r"https?://\S+\.(?:csv|tsv|jsonl?|parquet|arrow)(?:[?#]\S*)?(?:\s|$)"
    r"|(?:^|\s)[\w./-]+\.(?:csv|tsv|jsonl?|parquet|arrow)(?:\s|$)",
    re.I,
)
_SYNTHETIC_REQUEST = re.compile(
    r"\b(?:synthetic|simulated|toy|generated)\b.{0,40}"
    r"\b(?:data|dataset|samples?|instances?)\b"
    r"|(?:合成|模拟|随机生成|人工生成)(?:数据|数据集|样本|实例)",
    re.I,
)
_SYNTHETIC_GEN = re.compile(
    r"make_blobs|make_classification|make_regression|make_moons|make_circles|synthetic"
    r"|randomly generat|default_rng|np\.random|numpy\.random|gaussian (blob|mixture|cluster)"
    r"|simulated data|合成数据|随机生成|人工生成",
    re.I,
)


def _synthetic_for_real_dataset_reason(design: dict, question: str) -> str | None:
    """Reject synthetic substitution for an explicitly requested data source.

    The question, not a built-in dataset-name registry, is authoritative. A user who explicitly
    asks for synthetic data keeps that intent even if the word ``dataset`` also appears.
    """
    if _SYNTHETIC_REQUEST.search(question):
        return None
    if not (_EXPLICIT_DATA_SOURCE.search(question) or _SUPPLIED_DATA_LOCATION.search(question)):
        return None
    instance = str(design.get("instance_spec") or "") + " " + str(design.get("task") or "")
    if _SYNTHETIC_GEN.search(instance):
        return ("the question requires an external or real data source but instance_spec "
                "GENERATES synthetic data (synthetic data must not silently replace the "
                "requested source)")
    return None


def _fix_real_data_message(question: str, reason: str) -> str:
    return (
        f"User research question:\n{question}\n\n"
        f"Your previous design violates Real-Data-Only: {reason}. REDESIGN so instance_spec "
        "LOADS the actual requested data through its official loader or supplied location, with "
        "a real held-out split -- do NOT generate synthetic samples to represent it. If the data "
        "source genuinely cannot be loaded, say so explicitly rather than "
        "substituting synthetic data. Return the single JSON object with the required fields."
    )


def _narrow_message(question: str, reason: str) -> str:
    return (
        f"User research question:\n{question}\n\n"
        f"Your previous design was too broad: {reason}. REDESIGN it as EXACTLY ONE "
        "proposed method vs EXACTLY ONE standard baseline, on ONE scalar metric, at a "
        "single modest scale. If the question is comparative/broad, pick the SINGLE most "
        "representative one-vs-one comparison and scope everything else out (mention the "
        "rest only as future work, NOT in this experiment). The 'baseline' field must "
        "name exactly one method. Return the single JSON object with the required fields."
    )


# --- LLM design-faithfulness critic (P2) -------------------------------------------
# The regex guards above catch only OBVIOUS cases (literal "optimal", 3+ baselines).
# A DOMAIN-optimal baseline (WSPT/Smith's rule for weighted completion time, Dijkstra,
# Huffman, ...), a swapped method/baseline, an inverted direction, or an intractable
# instance spec need JUDGEMENT that no regex can encode. This LLM critic reviews the
# design against the user's question and, if unfaithful, drives ONE corrective
# re-design. Leak-safe: it sees only the question + the design's PUBLIC fields (it runs
# BEFORE the harness/method split, so there are no ruler internals to leak). Fail-OPEN:
# any critic error returns "faithful" so a flaky critic never blocks a real design.
_CRITIC_SYSTEM = (
    "You are a STRICT reviewer. Decide whether a proposed experiment DESIGN faithfully "
    "and tractably tests the USER'S research question. Check ALL of:\n"
    "1. FAITHFUL METHOD+BASELINE: the design tests the SAME proposed method the user "
    "asked about, against the SAME baseline the user named -- it must NOT swap their "
    "roles nor substitute a different method/baseline than the question specifies.\n"
    "2. BEATABLE BASELINE: the baseline must be an ordinary reference the proposed method "
    "could PLAUSIBLY beat -- NOT the exact optimum, an oracle, or an algorithm that is "
    "PROVABLY OPTIMAL for this metric (e.g. exact DP, an ILP/MILP/exhaustive solver, or a "
    "domain-optimal rule such as WSPT / Smith's rule for weighted completion time, "
    "Dijkstra for shortest paths, Huffman for prefix codes, or earliest-finish-time greedy "
    "for activity selection / interval scheduling). Nothing beats the optimum, so an optimal "
    "baseline makes the experiment unwinnable and meaningless. ALSO flag these DISGUISED "
    "forms: (a) the METRIC is a GAP / DISTANCE / REGRET / RATIO relative to the optimal (or "
    "to a reference the baseline attains) -- then the baseline scores 0 (or 1) BY DEFINITION "
    "and is unbeatable, even with no 'optimal' keyword; (b) the baseline is the best-known / "
    "provably-correct algorithm for the task. FIX: when the natural best answer IS a specific "
    "optimal/best algorithm, it must be the PROPOSED METHOD (compared against a SIMPLE "
    "beatable baseline -- naive / random / input-order) on a DIRECT metric (the actual "
    "quantity, e.g. count of intervals), NOT the baseline of a gap-from-optimal metric.\n"
    "3. CORRECT DIRECTION: success_criterion must say the PROPOSED METHOD achieves a "
    "BETTER metric than the baseline (in the metric's stated direction), not the reverse. "
    "FLAG any success_criterion that EXPECTS the method to be WORSE than the baseline, or "
    "frames success as 'characterizing / measuring how much WORSE the method is' -- that is "
    "a role inversion: the method is set up to lose and can never win.\n"
    "4. TRACTABLE, WELL-DEFINED INSTANCES: instance_spec must be a SIMPLE, unambiguously "
    "defined, seed-reproducible generator. Flag exotic/over-clever or ambiguous "
    "constructions that are hard to define or compute.\n"
    "5. ONE-VS-ONE: exactly one proposed method vs one baseline on one scalar metric, "
    "not a multi-method survey.\n"
    "6. FAIR COMPARISON: the method and the baseline must each get a configuration that "
    "gives it a genuine, fair chance. When the experiment isolates ONE variable (e.g. "
    "warmup on/off, optimizer A vs B, algorithm X vs Y), EVERY OTHER setting must be held "
    "at a value that is SENSIBLE FOR BOTH sides -- never a single shared hyperparameter "
    "that cripples one side (a classic mistake: reusing one learning rate for both Adam "
    "and SGD when that rate diverges for one of them), never an unequal compute/time/"
    "iteration budget, and never settings hand-tuned to favour the proposed method. Each "
    "side should get the reasonable, separately-appropriate settings it needs to perform "
    "well; only the variable under study may differ. Flag any rigged or apples-to-oranges "
    "comparison.\n"
    "7. DATA-SOURCE FIDELITY: if the user names or supplies a dataset, corpus, benchmark "
    "data source, loader, file or URL, the design must use that source. It must not silently "
    "replace requested real/external data with generated synthetic data. Synthetic data is "
    "fine only when the question asks for it or leaves the data source open.\n\n"
    "Respond with ONE JSON object: {\"faithful\": <bool>, \"issues\": [<short specific "
    "problem strings>]}. Set faithful=true ONLY if ALL checks pass; otherwise list the "
    "concrete issues. Be strict, but do NOT invent problems that are not there."
)


def _design_critic(llm, question: str, design: dict) -> dict:
    """LLM review of the design's faithfulness to ``question``. Returns
    ``{"faithful": bool, "issues": [str, ...]}``. Leak-safe (only the question + public
    design fields) and FAIL-OPEN (any error -> faithful, so it never blocks a design)."""
    fields = {k: design.get(k) for k in (
        "task", "baseline", "method_interface", "metric", "success_criterion",
        "instance_spec", "lower_is_better")}
    user = (f"User research question:\n{question}\n\n"
            f"Proposed design (JSON):\n{json.dumps(fields, ensure_ascii=False)}")
    try:
        result = llm.chat(
            messages=[{"role": "system", "content": _CRITIC_SYSTEM},
                      {"role": "user", "content": user}],
            response_format="json_object")
        obj = json.loads(extract_json(result.content or ""))
    except Exception:  # noqa: BLE001 -- advisory gate: a flaky critic must never block design
        return {"faithful": True, "issues": []}
    if not isinstance(obj, dict):
        return {"faithful": True, "issues": []}
    raw = obj.get("issues")
    issues = [str(i) for i in raw if str(i).strip()] if isinstance(raw, list) else []
    faithful = obj.get("faithful")
    faithful = bool(faithful) if isinstance(faithful, bool) else (not issues)
    return {"faithful": faithful, "issues": issues}


def _refaithful_message(question: str, issues: list) -> str:
    bullets = "\n".join(f"  - {i}" for i in issues)
    return (
        f"User research question:\n{question}\n\n"
        "Your previous design did NOT faithfully/tractably test this question. Fix ALL of:\n"
        f"{bullets}\n"
        "REDESIGN so it tests the METHOD the user proposed against the BASELINE the user "
        "named (same roles, correct direction), the baseline is BEATABLE (never the "
        "optimum or a provably-optimal algorithm for this metric), the comparison is FAIR "
        "(each side gets sensible, separately-appropriate settings -- no shared "
        "hyperparameter that cripples one side, no unequal budget, no tuning that favours "
        "the method; only the variable under study differs), and the instances are SIMPLE "
        "and unambiguously defined. If the natural best answer IS a specific optimal/best "
        "algorithm, make THAT algorithm the PROPOSED METHOD and compare it against a SIMPLE "
        "BEATABLE baseline (naive / random / input-order) on a DIRECT metric (the actual "
        "quantity, e.g. count of intervals scheduled -- NOT a gap/regret/ratio relative to "
        "the optimum, which makes the baseline unbeatable). Return the single JSON object "
        "with the required fields."
    )


def _design_once(llm, user_message: str, max_attempts: int) -> dict:
    """One design request with bounded retries on MALFORMED JSON (transient LLM
    hiccup). A well-formed-but-incomplete design raises immediately."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    last_exc: Exception | None = None
    for _attempt in range(max(1, max_attempts)):
        result = llm.chat(messages=messages, response_format="json_object")
        content = result.content or ""
        try:
            obj = json.loads(extract_json(content))
        except (json.JSONDecodeError, TypeError) as exc:
            last_exc = exc  # transient malformed JSON -> retry
            continue
        if not isinstance(obj, dict):
            raise DesignError("design JSON is not an object")
        missing = [f for f in REQUIRED_FIELDS if not obj.get(f)]
        if missing:
            raise DesignError(f"design is missing required field(s): {missing}")
        # dependencies is OPTIONAL and backward-safe: a missing / non-list value
        # coerces to [] (pure stdlib) rather than failing the design.
        deps = obj.get("dependencies")
        obj["dependencies"] = deps if isinstance(deps, list) else []
        # io_contract is OPTIONAL: a concrete shape example for check_shape. A
        # missing / non-dict value coerces to None (shape-check then no-ops).
        ioc = obj.get("io_contract")
        obj["io_contract"] = ioc if isinstance(ioc, dict) else None
        # lower_is_better decides the verdict's metric DIRECTION (a higher-is-better
        # metric like knapsack value would otherwise be INVERTED). Prefer the LLM's
        # boolean; fall back to inferring from the metric/success_criterion text.
        lib = obj.get("lower_is_better")
        obj["lower_is_better"] = lib if isinstance(lib, bool) else _infer_lower_is_better(obj)
        # gpu (training experiments run the scored step on a GPU node) -> bool, default False.
        obj["gpu"] = bool(obj.get("gpu"))
        return obj
    raise DesignError(
        f"LLM did not return valid JSON for the design after {max_attempts} attempts: {last_exc}"
    ) from last_exc


def design_experiment(llm, question: str, *, max_attempts: int = 3) -> dict:
    """Return a validated ExperimentDesign dict for ``question``.

    Retries on malformed JSON (transient). Then, if the design reads as a
    multi-method survey (over-scoped -> hard to produce ONE valid solution),
    issues ONE narrowing re-design to force ONE method vs ONE baseline. That
    re-design is bounded and accepted as-is (never loops / never hard-fails the
    run on a scope heuristic).
    """
    obj = _design_once(llm, _user_message(question), max_attempts)
    reason = _overscoped_reason(obj)
    if reason:
        obj = _design_once(llm, _narrow_message(question, reason), max_attempts)
    # ...and if the baseline is an unbeatable exact-optimum/oracle (method could never
    # beat it -> the run is meaningless), issue ONE corrective re-design. Bounded +
    # accepted as-is (never loops / never hard-fails on the heuristic).
    reason = _unbeatable_baseline_reason(obj)
    if reason:
        obj = _design_once(llm, _fix_baseline_message(question, reason), max_attempts)
    # LLM faithfulness critic (P2): the GENERAL net over the cheap regex guards above --
    # it catches domain-optimal baselines (WSPT/Dijkstra/Huffman...), swapped
    # method/baseline roles, an inverted direction, and intractable instances that no
    # regex can. If it flags issues, issue ONE corrective re-design (bounded, fail-open,
    # accepted as-is -- never loops / never hard-fails on the critic's opinion).
    review = _design_critic(llm, question, obj)
    if not review["faithful"] and review["issues"]:
        obj = _design_once(llm, _refaithful_message(question, review["issues"]), max_attempts)
    # DETERMINISTIC role-inversion floor (code-enforced): the LLM critic above is fail-open and
    # sometimes misses a success_criterion that frames the method as suboptimal/worse. This
    # guard cannot be skipped -- if it fires, issue one bounded corrective re-design.
    reason = _method_set_to_lose_reason(obj)
    if reason:
        obj = _design_once(llm, _fix_role_inversion_message(question, reason), max_attempts)
    # T6: an ML/statistical design must measure its metric on a HELD-OUT split (train-only
    # metrics overfit -> the evaluator flags "missing held_out validation"). Deterministic,
    # ML-metric-only, one bounded corrective re-design.
    reason = _missing_heldout_reason(obj)
    if reason:
        obj = _design_once(llm, _fix_heldout_message(question, reason), max_attempts)
    # T7 (Real Data Only): a question that names a real public dataset must not be answered with
    # synthetic data silently substituted for it. Deterministic, one bounded corrective re-design.
    reason = _synthetic_for_real_dataset_reason(obj, question)
    if reason:
        obj = _design_once(llm, _fix_real_data_message(question, reason), max_attempts)
    return obj


__all__ = [
    "design_experiment",
    "DesignError",
    "REQUIRED_FIELDS",
    "sanitize_dependencies",
    "pip_install_command",
]
