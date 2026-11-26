#!/usr/bin/env python
"""Opt-in REAL smoke test for OpenHandsBackend. NOT part of the pytest suite.

This starts a REAL Docker container and makes REAL OpenRouter LLM calls.
Run MANUALLY, SERIAL, single-session only:

    AUTORESEARCH_REAL_SANDBOX=1 OPENROUTER_KEY=sk-... \
        python scripts/smoke_openhands_backend.py

Flow: open a DockerWorkspace -> have the agent write metric.py + result.json ->
run it and hard-judge exit code / stdout -> export the workspace -> teardown.

NOTE (version compat): host SDK is 1.23.0 but the default image tag is
agent-server:1.19.1-python -- these are NOT guaranteed compatible. If the
handshake fails, bump AUTORESEARCH_SANDBOX_IMAGE to a matching tag (or pull one)
and rerun.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 清代理 env:本地 clash 代理会破坏 host<->docker 与容器内 OpenRouter 调用
# (vibe 的 clear_proxy_env 同因)。
for _var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_var, None)

# 让脚本能直接从仓库根运行(scripts/ 的父目录)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend  # noqa: E402

TASK = (
    "在 /workspace 下写 metric.py:计算 2+2、打印 'METRIC=4'、"
    '并把 {"metric_value": 4} 写进 result.json'
)


def main() -> int:
    if not os.environ.get("OPENROUTER_KEY"):
        print("OPENROUTER_KEY not set; export it and rerun. Aborting.")
        return 2
    if os.environ.get("AUTORESEARCH_REAL_SANDBOX") != "1":
        print("Set AUTORESEARCH_REAL_SANDBOX=1 to run the REAL smoke. Aborting.")
        return 2

    b = OpenHandsBackend()
    h = b.open_sandbox()
    try:
        print("[smoke] running agent task...")
        run = b.run_agent_task(h, TASK)
        print(f"[smoke] agent status={run.status} cost={run.cost}")

        print("[smoke] executing python metric.py ...")
        r = b.execute_command(h, "python metric.py")
        print(f"[smoke] exit_code={r.exit_code}")
        print(f"[smoke] stdout:\n{r.stdout}")
        assert r.exit_code == 0, f"expected exit 0, got {r.exit_code}"
        assert "METRIC=4" in r.stdout, f"stdout missing METRIC=4: {r.stdout!r}"

        dest = "/tmp/autoresearch-oh-smoke"
        b.export_workspace(h, dest)
        exported = sorted(str(p) for p in Path(dest).rglob("*") if p.is_file())
        print(f"[smoke] exported files:\n" + "\n".join(exported))
        rj = Path(dest) / "result.json"
        if rj.is_file():
            print(f"[smoke] result.json: {json.loads(rj.read_text())}")
        print("[smoke] OK")
        return 0
    finally:
        b.close_sandbox(h)


if __name__ == "__main__":
    raise SystemExit(main())
