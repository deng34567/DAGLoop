"""Per-run SEARCH experience memory (借鉴 Frontis-MA1/OpenMLE 的 experience cards + board).

setup_memory 记的是"这个仓库怎么装/怎么跑"(跨 session 复用);这里记的是"这一次研究里,搜索
过程试过哪些方法族、每族最好多少、哪些错反复出现"——即**单次运行内**的搜索足迹,喂给:
* 换方法时("这些方向试过且没赢,别再兜圈子");
* 重试/Debug 时("这个报错以前是这么处理的")。

三条纪律,同 setup_memory:
* **坏了不能拖垮主流程**:纯文件缓存,缺失/损坏 -> 当作"没有记忆",绝不抛。
* **只记确定性事实**:一张卡是一轮的客观结果(指标/是否有效/判决/错误签名),不写猜测。
* **签名要稳**:错误文本里塞满了路径/行号/地址/uuid,直接当 key 会每次都不一样;normalize_error
  把这些抹掉,只留"异常类型 + 归一化的核心信息",让"同一类错"能聚到一起。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# 归一化正则:抹掉每次都变的成分,让"同一类错"产生同一个签名。
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_PATH_RE = re.compile(r"(/[^\s\"'`:;|&()]+)+")          # 绝对路径
_LINENO_RE = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WS_RE = re.compile(r"\s+")
_EXC_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")


def normalize_error(text) -> str:
    """把原始错误文本压成稳定的"错误签名"(同一类错 -> 同一个字符串)。

    抹掉 hex/uuid/绝对路径/行号/数字等每次都变的成分,保留异常类型 + 归一化核心。空 -> ""。
    """
    if not text:
        return ""
    s = str(text).strip()
    # 只取首行的核心(traceback 末行/异常行信息量最大,但我们拿到的多是单行 "Type: msg")
    first = s.splitlines()[0] if s else ""
    exc = ""
    m = _EXC_RE.search(s)                 # 异常类型在全文里找(可能不在首行),原样保留大小写
    if m:
        exc = m.group(1)
    core = first
    if exc:                               # 去掉首行开头的 "ExcType:" 前缀,避免签名里重复
        core = re.sub(rf"^\s*{re.escape(exc)}\s*:?\s*", "", core)
    core = _UUID_RE.sub("<id>", core)
    core = _HEX_RE.sub("<hex>", core)
    core = _PATH_RE.sub("<path>", core)
    core = _LINENO_RE.sub("line <n>", core)
    core = _NUM_RE.sub("<n>", core)
    core = _WS_RE.sub(" ", core).strip().lower()
    core = core[:160]                     # 封顶,别让一条长信息撑爆
    sig = f"{exc}: {core}" if exc else core
    return sig[:200]


def is_infrastructure_failure(text) -> bool:
    """Return whether a failure belongs to the agent transport, not the experiment."""
    normalized = str(text or "").strip().lower()
    return (
        "conversation run failed" in normalized
        or "sandbox infra error" in normalized
        or normalized.startswith("infra:")
    )


@dataclass
class ExperienceCard:
    """一轮搜索的客观结果。operator/method_family 允许为空(未知时)。"""
    round: int
    operator: str = ""                    # draft/improve/debug/check-expectation 或 ""
    method_family: str = ""               # 方法族标签(如 "lr-schedule"、"tokenizer");""=未知
    metric_value: float | None = None
    valid: bool | None = None
    verdict: str = ""                     # new_best/valid_not_best/invalid/fail...
    error_signature: str = ""             # 失败时的 normalize_error 结果;成功为 ""
    lower_is_better: bool = True
    diff: str = ""                        # 这一轮 agent 对仓库的实际代码改动(git diff);P0 累积/真Crossover用(远程best-effort)
    changes_summary: str = ""             # agent 自报:本轮具体改了啥(比 git diff 远程更可靠)
    rationale: str = ""                   # agent 自报:为什么这么改(假设/预期)——方法反思回路输入
    bottleneck: str = ""                  # agent 自报:本轮后当前性能瓶颈/哪里没达预期——下一轮/反思攻的点
    config: str = ""                      # agent 自报:本轮固定的实验配置(model尺寸/iters/data/eval)。第1轮的用作跨轮锁定基准
    ablation_report: str = ""             # check-expectation 轮的逐组件消融报告(JSON字符串);#84 由沙箱产出、handler 落盘
    implementation_verdict: str = ""      # faithful/partial/...; partial 核心可继续逐组件 improve
    repo_path: str = ""                   # 本轮完整工作树;下轮直接继承,不靠截断 diff 重建
    component_id: str = ""                # ComponentPlan 当前节点；空表示 legacy/整体验证轮
    parent_checkpoint: str = ""            # 当前组件实际继承的 checkpoint
    component_status: str = ""             # accepted/execution_failed/implementation_failed/...

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperienceCard":
        allowed = {f for f in cls.__dataclass_fields__}          # 忽略未知字段,向后兼容
        return cls(**{k: v for k, v in (d or {}).items() if k in allowed})


class SearchBoard:
    """单次运行的搜索足迹,JSON 落盘(结构 {"cards": [...]}).缺失/损坏 -> 空,永不抛。"""

    def __init__(self, path: str) -> None:
        self.path = path

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {"cards": []}
        except (OSError, json.JSONDecodeError):
            return {"cards": []}

    def _save(self, data: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)    # 原子:写一半崩了也不会污染
        except OSError:
            logger.warning("could not persist search memory to %s", self.path, exc_info=True)

    def add_card(self, card: ExperienceCard) -> None:
        data = self._load()
        data.setdefault("cards", []).append(card.to_dict())
        self._save(data)

    def remember_rejected_method(self, method: dict) -> None:
        """Persist a falsified whole-method definition across later regeneration hops."""
        if not isinstance(method, dict) or not str(method.get("approach") or "").strip():
            return
        canonical = json.dumps(method, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        import hashlib
        fingerprint = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        data = self._load()
        rejected = data.setdefault("rejected_methods", [])
        if any(item.get("fingerprint") == fingerprint for item in rejected
               if isinstance(item, dict)):
            return
        rejected.append({"fingerprint": fingerprint, "method": method})
        self._save(data)

    def begin_method(self, fingerprint: str) -> None:
        """Start an operator lineage for one concrete method definition.

        A revised method must not inherit the prior method's best diff.  Keep those cards as
        session history for later whole-method regeneration, while exposing an empty current board
        to ``decide_operator``.  Replaying the same generated method is idempotent.
        """
        marker = str(fingerprint or "").strip()
        if not marker:
            return
        data = self._load()
        if data.get("active_method_fingerprint") == marker:
            return
        current = data.get("cards")
        if isinstance(current, list) and current:
            data.setdefault("history_cards", []).extend(current)
        data["cards"] = []
        data["active_method_fingerprint"] = marker
        self._save(data)

    def cards(self) -> list[ExperienceCard]:
        return [ExperienceCard.from_dict(c) for c in self._load().get("cards", [])]

    def all_cards(self) -> list[ExperienceCard]:
        """Archived prior-method cards followed by the current method's cards."""
        data = self._load()
        raw = list(data.get("history_cards", [])) + list(data.get("cards", []))
        return [ExperienceCard.from_dict(c) for c in raw]

    # 聚合方法都接受可选 cards(P1-2 省开销:一次决策只读盘一次,不各自重复 _load)。
    def method_families_tried(self, cards: list | None = None) -> list[str]:
        """去重、保序的已试方法族(排除空标签)。"""
        cards = self.cards() if cards is None else cards
        seen, out = set(), []
        for c in cards:
            fam = (c.method_family or "").strip()
            if fam and fam not in seen:
                seen.add(fam)
                out.append(fam)
        return out

    def best_per_family(self, cards: list | None = None) -> dict[str, ExperienceCard]:
        """每个方法族里指标最好的一张有效卡(尊重 lower_is_better 方向)。"""
        cards = self.cards() if cards is None else cards
        best: dict[str, ExperienceCard] = {}
        for c in cards:
            fam = (c.method_family or "").strip()
            if not fam or c.metric_value is None or c.valid is False:
                continue
            cur = best.get(fam)
            if cur is None or (
                c.metric_value < cur.metric_value if c.lower_is_better
                else c.metric_value > cur.metric_value
            ):
                best[fam] = c
        return best

    def failure_patterns(self, cards: list | None = None) -> list[tuple[str, int]]:
        """反复出现的错误签名,按出现次数降序。"""
        cards = self.cards() if cards is None else cards
        counts: dict[str, int] = {}
        for c in cards:
            sig = (c.error_signature or "").strip()
            if sig:
                counts[sig] = counts.get(sig, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    def by_error_signature(self, signature: str, cards: list | None = None) -> list[ExperienceCard]:
        """历史上撞过同一错误签名的卡(给 Debug 复用之前的处理上下文)。"""
        sig = (signature or "").strip()
        if not sig:
            return []
        cards = self.cards() if cards is None else cards
        return [c for c in cards if (c.error_signature or "").strip() == sig]

    def summarize_for_regen(self, *, max_families: int = 8, max_failures: int = 3) -> str:
        """给"换方法"喂的一句话:试过的方法族(带最好指标)+ 反复出现的失败模式。无则空串。
        P1-2:一次读盘,把 cards 传给各聚合,不重复 _load;并对族/失败数量封顶,不撑爆 prompt。"""
        data = self._load()
        raw_cards = list(data.get("history_cards", [])) + list(data.get("cards", []))
        cards = [ExperienceCard.from_dict(c) for c in raw_cards]
        parts: list[str] = []
        rejected = []
        for item in data.get("rejected_methods", [])[:max_families]:
            method = item.get("method") if isinstance(item, dict) else None
            approach = str((method or {}).get("approach") or "").strip()
            if approach:
                rejected.append(approach[:600])
        if rejected:
            parts.append("已淘汰方法定义(不得回退): " + " || ".join(rejected))
        best = self.best_per_family(cards)
        fams = self.method_families_tried(cards)[:max_families]
        if fams:
            items = []
            for fam in fams:
                b = best.get(fam)
                items.append(f"{fam}(best={b.metric_value})" if b else f"{fam}(无有效结果)")
            parts.append("已试方法族: " + "; ".join(items))
        fails = self.failure_patterns(cards)[:max_failures]
        if fails:
            parts.append("反复出现的失败: " + "; ".join(f"{sig}(×{n})" for sig, n in fails))
        note = self._latest_reflection_method_note()   # C-4: 反思判定方法层要改 -> 喂进换/改 method
        if note:
            parts.append(f"反思判定方法层需修订: {note}")
        return " | ".join(parts)

    def _latest_reflection_method_note(self) -> str:
        """读同目录 reflection_reports.jsonl 的最后一条;若最近一次反思 route=revise_method,返回其
        method_note(供"换/改 method"知悉方法哪里错了)。无/异常 -> ""(best-effort)。"""
        try:
            rp = os.path.join(os.path.dirname(self.path), "reflection_reports.jsonl")
            if not os.path.isfile(rp):
                return ""
            last = ""
            with open(rp, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        last = line
            if not last:
                return ""
            obj = json.loads(last)
            if obj.get("route") == "revise_method":
                return str(obj.get("method_note") or "")[:800]
        except Exception:  # noqa: BLE001
            pass
        return ""


def _diff_max() -> int:
    """每段 diff 喂给 agent 的字符上限(防撑爆 prompt)。env AUTORESEARCH_OPERATOR_DIFF_MAX,默认4000。"""
    try:
        return max(200, int(os.environ.get("AUTORESEARCH_OPERATOR_DIFF_MAX", "4000")))
    except (TypeError, ValueError):
        return 4000


def _diff_block(label: str, diff: str) -> str:
    """把一段真实 git diff 包成算子提示里的一个代码块(截断,防撑爆 prompt)。空 diff -> 空串。"""
    d = (diff or "").strip()
    if not d:
        return ""
    cap = _diff_max()
    if len(d) > cap:
        d = d[:cap] + "\n…(diff 已截断,只给了前一部分)"
    return f"\n{label}:\n```diff\n{d}\n```\n"


def _change_block(label: str, card: "ExperienceCard") -> str:
    """一张卡的完整"改动叙述"给算子:agent 自报的 changes_summary(远程可靠、主) + git diff(有才带,
    补充) + 当时的 rationale + 它指出的 bottleneck。全空 -> 空串。反思回路让算子"有方向"的载体。"""
    cap = _diff_max()
    parts: list[str] = []
    cs = (getattr(card, "changes_summary", "") or "").strip()
    if cs:
        parts.append(f"改动摘要: {cs[:cap]}")
    dblock = _diff_block("实际 diff", card.diff)      # 保留其截断("diff 已截断")与 ```diff``` 渲染
    if dblock:
        parts.append(dblock.strip())
    r = (getattr(card, "rationale", "") or "").strip()
    if r:
        parts.append(f"当时的理由: {r[:800]}")
    b = (getattr(card, "bottleneck", "") or "").strip()
    if b:
        parts.append(f"它指出的瓶颈: {b[:800]}")
    if not parts:
        return ""
    return f"\n{label}:\n" + "\n".join(parts) + "\n"


def decide_operator(board: "SearchBoard") -> tuple[str, str]:
    """借鉴 Frontis-MA1 的 Draft/Improve/Debug 算子分型:看这次运行的搜索足迹,决定
    下一轮该用哪个算子,并给出一段可直接前置到实验请求里的"策略提示"(operator-conditioned
    context)。优先级:上一轮失败 -> Debug;已有可用变体 -> Improve;否则 Draft。
    Crossover 始终失活：顺序 improve 已经累积了先前的最优改动。

    返回 (operator, preamble)。preamble 为 "" 表示 draft(不改变原有行为)。纯函数,不做 IO 决策
    以外的事;调用方 fail-safe 包裹。
    """
    cards = [
        card for card in board.cards()
        if not is_infrastructure_failure(card.error_signature)
    ]                                        # P1-2: 一次读盘,后续全在内存里算
    if not cards:
        return ("draft", "")
    last = cards[-1]
    lower = bool(last.lower_is_better)

    # Debug:上一轮以错误失败 -> 先修这个错,并带上"这个错以前撞过几次" + 失败那轮的实际改动。
    if last.valid is False and last.error_signature:
        seen = len(board.by_error_signature(last.error_signature, cards))
        ctx = (f"[本轮算子=DEBUG] 上一轮尝试以错误失败: “{last.error_signature}”"
               f"(该类错误已出现 {seen} 次)。先诊断并修复这个失败,再谈改进;"
               "避免重复触发同一个错误。" + _change_block("上一轮失败的改动(在此基础上定向修)", last))
        return ("debug", ctx)

    # The method prompt defines DRAFT -> IMPROVE as progressive construction.  A measured,
    # grounded partial core is progress, not an execution failure or a whole-method ablation.
    if last.valid is True and last.implementation_verdict == "partial":
        change = _change_block(
            "已跑通的部分实现(保留它,再补下一个确认组件)", last)
        return (
            "improve",
            "[本轮算子=IMPROVE] 上一轮已经真实跑通确认方法的核心，但实现审查仍为 partial。"
            "保留已有实现和数值，优先引入下一个尚未实现的组件；不得把 partial 误当执行错误，"
            "也不要提前做全方法消融。" + change,
        )

    if last.valid is True and last.implementation_verdict == "drifted":
        change = _change_block(
            "已测得有效数值、但偏离确认方法的改动(保留作比较锚点)", last)
        return (
            "improve",
            f"[本轮算子=IMPROVE] 上一轮实际代码偏离了确认方法，但它产生了真实可测数值 "
            f"{last.metric_value}。该数值只能称为 off-method observation，不能冒充确认方法结果。"
            "保留它的改动、工作树和数值作为经验与比较锚点；下一轮优先把实现向确认方法靠拢，"
            "并同时与 frozen baseline 和该观测值比较。若证据表明真正有效的机制与确认方法不兼容，"
            "不要在沙箱内悄悄改名；记录证据，交由正式 RevisingMethod 修订方法描述。" + change,
        )

    # Check-Expectation(会议 2026-08-11 · 实验层消融钩子):上一轮【跑通了但没涨点】,或【跑了没崩
    # 却没拿到可用结果】——这时不知道是"方法本身不兑现预期"还是"实现细节的问题"。进"查预期"算子:
    # 照【确认方法的组件契约】逐组件跑判据(直接增益→加/减测 metric；间接增益→测中间性质),
    # 客观判每个组件是否符合预期。防重入:上一轮已是 check-expectation
    # 就不再连着查(其 report 由 handler/#84 消费路由:有组件 unmet→回方法层重设计;全符合→improve)。
    _not_gain = (last.valid is True and last.verdict == "valid_not_best") or \
                (last.valid is False and not last.error_signature)
    if _not_gain and last.operator != "check-expectation":
        ctx = ("[本轮算子=CHECK-EXPECTATION] 上一轮没有达到预期(没涨点/没拿到可用结果)。先【不改方法】,"
               "对【已确认方法的每个已声明组件】逐一按其契约做定向验证,不得只挑其中 1-2 个或漏项:"
               "直接增益组件→按其 ablation_test 加上/去掉该组件,测 target metric 的实际变化;"
               "间接增益组件→按其验收契约测中间性质(如正交度或解耦度)。"
               "结构化报告 ABLATION_REPORT:每个组件 {component, type, actual(有限数字), "
               "expected:{operator:<|<=|>|>=|==, value:有限数字}, evidence:{source, ref}, "
               "verdict}。主机会从 actual/expected 重算结果，不信任自报 verdict；证据不全则只能"
               "low_confidence。这一步只诊断、不追求涨点。")
        return ("check-expectation", ctx)

    # P0: 拿"最优的几张有效卡"(带真实 diff),方向敏感、按值去重、最好在前。
    valid_cards = [c for c in cards if c.valid and c.metric_value is not None]
    valid_cards.sort(key=lambda c: c.metric_value, reverse=not lower)
    seen_val: set = set()
    best: list[ExperienceCard] = []
    for c in valid_cards:
        if c.metric_value not in seen_val:
            seen_val.add(c.metric_value)
            best.append(c)

    if best:
        # Improve:给 agent 上一轮最优的【真实 diff】,让它先应用再叠加(累积,不从头)。
        c = best[0]
        d = _change_block("上一轮最优的改动(先把它应用到代码里,再在此基础上继续改进)", c)
        start = d if d else "从既有 baseline 出发,"
        ctx = (f"[本轮算子=IMPROVE] 已有一个可用变体达到 {c.metric_value}。{start}"
               "优先从已确认方法中引入下一个尚未实现的组件；若其组件已全部实现，再围绕已定位瓶颈"
               "做最小改进。这一轮的目标是明确超过它，不要脱离确认方法做泛化调参。")
        return ("improve", ctx)
    return ("draft", "")


def prior_worktree_for_next_operator(board: "SearchBoard") -> str:
    """Return the exact worktree that the next Debug/Improve round should inherit."""
    cards = [
        card for card in board.cards()
        if not is_infrastructure_failure(card.error_signature)
    ]
    if not cards:
        return ""
    last = cards[-1]
    if last.valid is False and last.error_signature and last.repo_path:
        return str(last.repo_path or "")
    if last.valid is True and last.implementation_verdict == "partial":
        return str(last.repo_path or "")
    if last.valid is True and last.implementation_verdict == "drifted":
        return str(last.repo_path or "")
    valid = [card for card in cards if card.valid and card.metric_value is not None]
    if not valid:
        return ""
    lower = bool(last.lower_is_better)
    valid.sort(key=lambda card: card.metric_value, reverse=not lower)
    best_metric = valid[0].metric_value
    # Recovery may append the persisted worktree for an already-recorded best metric.
    # Prefer that concrete tree over an older metric-identical card whose legacy record
    # predates ``repo_path``; otherwise the next attempt silently starts from a fresh clone.
    best_with_tree = next(
        (card for card in valid
         if card.metric_value == best_metric and card.repo_path),
        None,
    )
    return str((best_with_tree or valid[0]).repo_path or "")


def check_expectation(ablation_report: list) -> dict:
    """会议 2026-08-11 · 实验层消融钩子的【主机侧路由】纯函数。

    ``ablation_report`` 中每项必须有有限数字 ``actual``、结构化
    ``expected={operator,value}`` 与 ``evidence={source,ref}``。主机独立重算阈值判定；
    沙箱的 ``verdict`` 只是记录，不能改写 actual/expected 的结果。结构或证据不全则
    ``low_confidence``，不用于方法层归因。
    路由: 有 unmet -> route=revise_method(带 failing 组件, 供 MOSS component_ablation_failed 定位);
          否则 -> route=improve。额外返回 direct/indirect gain 组件列表，供编排层判断“局部声称满足但
          整体不涨点”的方法级矛盾。
    返回 {"route", "failing": [component...], "low_confidence": [component...], "summary"}。"""
    failing, low_conf, met = [], [], []
    gain_components, functional_components = [], []
    for item in ablation_report or []:
        if not isinstance(item, dict):
            continue
        comp = str(item.get("component") or "").strip() or "(unnamed)"
        ctype = str(item.get("type") or "").strip().lower()
        ctype = {"gain": "direct_gain", "functional": "indirect_gain"}.get(ctype, ctype)
        actual = item.get("actual")
        expected = item.get("expected")
        evidence = item.get("evidence")
        if ctype == "direct_gain":
            gain_components.append(comp)
        elif ctype == "indirect_gain":
            functional_components.append(comp)
        try:
            actual_value = float(actual)
            expected_value = float(expected["value"])
            operator = expected["operator"]
            structured = (
                not isinstance(actual, bool)
                and math.isfinite(actual_value)
                and not isinstance(expected["value"], bool)
                and math.isfinite(expected_value)
                and operator in {"<", "<=", ">", ">=", "=="}
                and isinstance(evidence, dict)
                and bool(str(evidence.get("source") or "").strip())
                and bool(str(evidence.get("ref") or "").strip())
            )
        except (KeyError, TypeError, ValueError):
            structured = False
            actual_value = expected_value = 0.0
            operator = ""
        if ctype not in ("direct_gain", "indirect_gain") or not structured:
            low_conf.append(comp)
        else:
            passed = {
                "<": actual_value < expected_value,
                "<=": actual_value <= expected_value,
                ">": actual_value > expected_value,
                ">=": actual_value >= expected_value,
                "==": actual_value == expected_value,
            }[operator]
            (met if passed else failing).append(comp)
    route = "revise_method" if failing else "improve"
    if failing:
        summary = "组件未兑现预期(回方法层重设计): " + "、".join(failing)
    elif low_conf:
        summary = "判据证据不足，不能声称符合；不阻塞但需补测: " + "、".join(low_conf)
    else:
        summary = "全部组件符合预期(含涨了没到量级的部分符合)"
    return {"route": route, "failing": failing, "low_confidence": low_conf,
            "met": met, "gain_components": gain_components,
            "functional_components": functional_components, "summary": summary}


DEFAULT_BASENAME = "search_memory.json"


def board_for_workroot(workroot: str) -> SearchBoard:
    """一次运行的 workroot -> 它的搜索足迹文件(与 improve 产物同目录)。"""
    return SearchBoard(os.path.join(workroot or ".", DEFAULT_BASENAME))


def session_workroot(session_id: str) -> str:
    """RunnerAgent 本地实验的 workroot 约定(与 handler._run_repo_experiment_local 一致),
    让写卡的 runner 和读卡的 method_agent 共用同一处路径,不各自拼。"""
    import tempfile
    base = os.environ.get("AUTORESEARCH_WEB_WORKDIR") or tempfile.gettempdir()
    return os.path.join(base, "repo_improve_local", session_id or "")


def board_for_session(session_id: str) -> SearchBoard:
    """按 session_id 定位该次运行的搜索足迹(跨 agent 读写同一份)。"""
    return board_for_workroot(session_workroot(session_id))


__all__ = [
    "normalize_error", "ExperienceCard", "SearchBoard", "decide_operator",
    "prior_worktree_for_next_operator",
    "board_for_workroot", "board_for_session", "session_workroot", "DEFAULT_BASENAME",
]
