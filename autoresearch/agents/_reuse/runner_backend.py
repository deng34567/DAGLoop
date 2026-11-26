"""
RunnerBackend: the execution-backend adapter interface for RunnerAgent.
RunnerAgent 的执行后端适配器接口。

RunnerAgent (AutoResearch state machine V1.10 §1) reproduces a benchmark's
reported baseline by running it through this adapter, then compares the
observed metric against the reported one (outside this module, in
runner_agent/core.py) to decide whether the baseline was reproduced. The
adapter returns a RawRunResult dict:
``{metric_value, environment, command, seed, runtime_seconds, exit_code,
log_excerpt}``.

Backends: FakeRunnerBackend (canned, in-memory -- the default in all mocked
tests, NEVER executes). E2BRunnerBackend is still an opt-in stub (raises).
LocalSubprocessRunnerBackend is a REAL, OPT-IN backend (P3): it runs a benchmark
in a SANDBOXED subprocess (temp workdir + rlimits + timeout + best-effort
network isolation) and reads the metric deterministically from result.json --
never the default; callers pass it explicitly (e.g. the web viewer with
AUTORESEARCH_REAL_RUNNER=1 on the bundled toy benchmark).

NO LLM imports here. Subprocess/tempfile are imported lazily inside
LocalSubprocessRunnerBackend so the module is import-light and Fake stays pure.
"""

from __future__ import annotations

from typing import Protocol


class RunnerBackend(Protocol):
    """Execution-backend adapter: runs a benchmark's baseline command.

    `benchmark_spec` and `verifier_spec` are the (dict-shaped) frozen
    BenchmarkSpec / VerifierSpec, passed through in case a backend wants to
    use the run command/environment/seed policy to shape execution;
    implementations are free to ignore fields they don't need.
    """

    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        """Run the baseline command and return a RawRunResult dict."""
        ...

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        """Run an experiment patch and return a RawRunResult dict.

        Unlike ``run_baseline``'s RawRunResult, this result MUST include
        ``stdout``/``stderr`` keys (RunnerAgent core's RunRecord requires a
        non-nullable ``stdout_excerpt``).
        """
        ...


_DEFAULT_RAW_RESULT: dict = {
    "metric_value": 0.5,
    "environment": "py3.11/cpu",
    "command": "python baseline.py",
    "seed": "0",
    "runtime_seconds": 1.0,
    "exit_code": 0,
    "log_excerpt": "baseline run completed successfully.",
}

_DEFAULT_EXPERIMENT_RAW_RESULT: dict = {
    "metric_value": 0.6,
    "environment": "py3.11/cpu",
    "seed": "0",
    "runtime_seconds": 2.0,
    "exit_code": 0,
    "stdout": "experiment run completed successfully.",
    "stderr": None,
}


class FakeRunnerBackend:
    """In-memory RunnerBackend for tests: returns a canned RawRunResult and
    records how many times it was called.
    测试用内存后端：返回预设的 RawRunResult，并记录调用次数。

    A canned ``raw_result`` dict may be passed directly, or built via the
    convenience kwargs (``metric_value``/``exit_code``/``command``/
    ``environment``), which override the corresponding field of the default
    successful run.
    """

    def __init__(
        self,
        *,
        raw_result: dict | None = None,
        metric_value: float | None = None,
        exit_code: int | None = None,
        command: str | None = None,
        environment: str | None = None,
        stdout: str | None = None,
    ) -> None:
        if raw_result is not None:
            self._raw_result = dict(raw_result)
        else:
            self._raw_result = dict(_DEFAULT_RAW_RESULT)
            if metric_value is not None:
                self._raw_result["metric_value"] = metric_value
            if exit_code is not None:
                self._raw_result["exit_code"] = exit_code
            if command is not None:
                self._raw_result["command"] = command
            if environment is not None:
                self._raw_result["environment"] = environment

        self._experiment_raw_result = dict(_DEFAULT_EXPERIMENT_RAW_RESULT)
        if metric_value is not None:
            self._experiment_raw_result["metric_value"] = metric_value
        if exit_code is not None:
            self._experiment_raw_result["exit_code"] = exit_code
        if environment is not None:
            self._experiment_raw_result["environment"] = environment
        if stdout is not None:
            self._experiment_raw_result["stdout"] = stdout

        self.calls = 0

    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        self.calls += 1
        return self._raw_result

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        self.calls += 1
        return self._experiment_raw_result


class E2BRunnerBackend:
    """Opt-in, NOT-wired real RunnerBackend stub (E2B sandboxed execution).
    Deferred past M10: every M10 test injects FakeRunnerBackend instead.
    Present only so the adapter interface is pluggable.

    DO NOT instantiate/run this in production or tests until a real backend
    is actually wired in a later milestone. NEVER runs a subprocess or
    network call.
    仅为占位/预留接口，M10 中不会被实际调用；真实实现推迟到后续里程碑。
    """

    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        raise NotImplementedError("real execution deferred / opt-in")

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        raise NotImplementedError("real execution deferred / opt-in")


_UNSHARE_OK: bool | None = None


def _unshare_available() -> bool:
    """Whether ``unshare -rn`` (unprivileged user+net namespace) actually works
    here -- best-effort network isolation for the sandbox. Cached."""
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        import shutil as _sh
        import subprocess as _sp
        _UNSHARE_OK = False
        if _sh.which("unshare"):
            try:
                r = _sp.run(["unshare", "-rn", "true"], capture_output=True, timeout=5)
                _UNSHARE_OK = (r.returncode == 0)
            except Exception:
                _UNSHARE_OK = False
    return _UNSHARE_OK


class LocalSubprocessRunnerBackend:
    """REAL, OPT-IN execution backend: runs the benchmark (baseline, or the
    experiment patch applied on top) in a SANDBOXED subprocess and reads the
    metric DETERMINISTICALLY from ``result.json`` (never trusts the LLM).

    Sandbox (best-effort, unprivileged): a fresh temp workdir, RLIMIT_AS (mem)
    + RLIMIT_CPU limits via preexec, a wall-clock ``timeout``, and network
    isolation via ``unshare -rn`` when available. This is adequate for the toy
    benchmark; it is NOT hardened for adversarial code -- Docker isolation is
    the deferred hardening (proposal D1). Verifier-first: a patch may only write
    files inside ``verifier_spec.editable_region`` (default-deny) and never
    ``forbidden_files``; path-escape is rejected.

    Never the default -- callers opt in explicitly by passing this backend.
    """

    def __init__(self, benchmark_dir, *, timeout: float = 60.0, mem_mb: int = 512,
                 run_cmd=("python", "run.py")) -> None:
        from pathlib import Path
        self.benchmark_dir = Path(benchmark_dir)
        self.timeout = float(timeout)
        self.mem_bytes = int(mem_mb) * 1024 * 1024
        self.run_cmd = list(run_cmd)

    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        r = self._run(None, verifier_spec)
        return {
            "metric_value": r["metric_value"], "environment": self._env(),
            "command": " ".join(self.run_cmd), "seed": "0",
            "runtime_seconds": r["runtime_seconds"], "exit_code": r["exit_code"],
            "log_excerpt": (r["log"] or "")[:500],
        }

    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        changes = (experiment_patch or {}).get("changes") or []
        r = self._run(changes, verifier_spec)
        return {
            "metric_value": r["metric_value"], "environment": self._env(), "seed": "0",
            "runtime_seconds": r["runtime_seconds"], "exit_code": r["exit_code"],
            "stdout": (r["log"] or "")[:2000], "stderr": (r["stderr"] or None),
        }

    # -- internals --
    @staticmethod
    def _env() -> str:
        import sys
        return f"py{sys.version_info.major}.{sys.version_info.minor}/cpu"

    def _run(self, patch_changes, verifier_spec) -> dict:
        import json
        import shutil
        import tempfile
        from pathlib import Path

        vs = verifier_spec or {}
        editable = set(vs.get("editable_region") or [])
        forbidden = set(vs.get("forbidden_files") or [])
        workdir = Path(tempfile.mkdtemp(prefix="ar-run-"))
        try:
            for p in self.benchmark_dir.iterdir():
                if p.is_file():
                    shutil.copy2(p, workdir / p.name)
            for ch in (patch_changes or []):
                f = (ch or {}).get("file")
                content = (ch or {}).get("patch", "")
                if not f:
                    continue
                if editable and f not in editable:
                    return self._rejected(f"file {f!r} not in editable_region (default-deny)")
                if f in forbidden:
                    return self._rejected(f"file {f!r} is forbidden")
                target = (workdir / f).resolve()
                if not str(target).startswith(str(workdir.resolve())):
                    return self._rejected(f"path escape rejected: {f!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            code, out, err, timed_out = self._sandboxed_run(workdir)
            metric, rt = 0.0, 0.0
            rj = workdir / "result.json"
            if rj.is_file():
                try:
                    d = json.loads(rj.read_text(encoding="utf-8"))
                    metric = float(d.get("metric_value", 0.0))
                    rt = float(d.get("runtime_seconds", 0.0))
                except Exception:
                    pass
            if timed_out:
                return {"metric_value": 0.0, "runtime_seconds": self.timeout,
                        "exit_code": -1, "log": (out or "") + "\n[TIMEOUT]", "stderr": "TIMEOUT"}
            return {"metric_value": metric, "runtime_seconds": rt,
                    "exit_code": code, "log": out, "stderr": err}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _rejected(msg: str) -> dict:
        return {"metric_value": 0.0, "runtime_seconds": 0.0, "exit_code": 126,
                "log": f"REJECTED: {msg}", "stderr": msg}

    def _sandboxed_run(self, workdir):
        import subprocess

        def _preexec():
            import resource
            for res, lim in ((resource.RLIMIT_AS, self.mem_bytes),
                             (resource.RLIMIT_CPU, int(self.timeout) + 5)):
                try:
                    resource.setrlimit(res, (lim, lim))
                except Exception:
                    pass

        cmd = (["unshare", "-rn"] + self.run_cmd) if _unshare_available() else list(self.run_cmd)
        try:
            p = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True,
                               timeout=self.timeout, preexec_fn=_preexec)
            return p.returncode, p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            return -1, (e.stdout or ""), "TIMEOUT", True


__all__ = [
    "RunnerBackend",
    "FakeRunnerBackend",
    "E2BRunnerBackend",
    "LocalSubprocessRunnerBackend",
]
