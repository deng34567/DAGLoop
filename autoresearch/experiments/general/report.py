"""Task 6: the honest experiment report (Markdown text).

``build_report(design, primary, reproduce, verdict) -> str`` renders the design,
the REAL baseline-vs-method numbers, the reproduce run, the isolation EVIDENCE,
the verdict -- and an explicit HONESTY note (N1): a ``pass`` means ONLY that an
AI-chosen method beat an AI-chosen baseline on an AI-chosen metric, REPRODUCIBLY
and ISOLATED. It does NOT mean the method is genuinely good, nor that the ruler
(the AI-written harness) is fair -- the ruler is AI-authored and UNAUDITED in P1;
adversarial review is deferred to P2.
"""
from __future__ import annotations

_HONESTY_NOTE = (
    "诚实声明(N1):**pass 仅代表**「一个 AI 选的方法,在一个 AI 选的基线、AI 选的"
    "指标上,可复现地、运行期隔离地赢了」——**不代表方法真的好,也不代表这把尺"
    "(harness)公平**。这把尺是 AI 现写、P1 未经对抗审查(留待 P2)。\n\n"
    "运行期隔离是**结构性强制**的(非文本猜测):可信编排器 `_orchestrator.py` 先 "
    "`import harness` 载入内存,随即把 `harness.py` 从磁盘删除,再经可信隔离器在独立"
    "子进程里跑 method——AI 写的 harness 只是被当库调用、从不运行 method,故 method "
    "碰不到标尺文件。**残余(留 P2)**:若某个极端对抗(合谋)的 harness 把真值/答案藏进 "
    "/workspace 下**任意文件**(含删后重建 harness.py、或直接经实例通道泄漏),当前尚未"
    "封堵;需要完整文件系统沙箱(bwrap/unshare/"
    "独立容器)才能彻底关闭。数字为真跑所得、可复现,但仍非「铁证」。"
)


def _fmt(v) -> str:
    return "?" if v is None else f"{v}"


def build_report(design: dict, primary: dict, reproduce: dict, verdict: dict) -> str:
    isolated_primary = bool(primary.get("isolated"))
    isolated_repro = bool(reproduce.get("isolated"))
    vmap = {"pass": "✅ 通过", "fail": "❌ 未通过", "inconclusive": "⚠️ 存疑"}
    vlabel = vmap.get(verdict.get("verdict"), verdict.get("verdict", "?"))

    lines = [
        "# 通用诚实实验报告 (P1)",
        "",
        "## 问题",
        str(design.get("question", "")),
        "",
        "## 实验设计",
        f"- 任务: {design.get('task')}",
        f"- 实例生成: {design.get('instance_spec')}",
        f"- 基线: {design.get('baseline')}",
        f"- 方法接口: {design.get('method_interface')}",
        f"- 指标: {design.get('metric')}",
        f"- 成功判据: {design.get('success_criterion')}",
        "",
        "## 真实数字",
        f"- 主跑 (seed {_fmt(primary.get('seed'))}): "
        f"基线={_fmt(primary.get('baseline_metric'))} vs "
        f"方法={_fmt(primary.get('method_metric'))} "
        f"(valid={primary.get('valid')})",
        f"- 复现 (seed {_fmt(reproduce.get('seed'))}): "
        f"基线={_fmt(reproduce.get('baseline_metric'))} vs "
        f"方法={_fmt(reproduce.get('method_metric'))} "
        f"(valid={reproduce.get('valid')})",
        "",
        "## 隔离证据",
        f"- 运行期隔离(结构性): 主跑={isolated_primary}, 复现={isolated_repro} "
        "(可信编排器载入 harness 后删除 harness.py、经隔离器跑 method;True=可信路径跑通,非文本猜测)",
        "",
        "## 验收判决",
        f"- 判决: {vlabel}",
        f"- 理由: {'; '.join(verdict.get('reasons', []))}",
        "",
        "## 诚实声明",
        _HONESTY_NOTE,
    ]
    return "\n".join(lines)


__all__ = ["build_report"]
