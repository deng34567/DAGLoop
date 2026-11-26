"""Remote GPU runner: run one experiment on a real GPU node via the SAME trusted
orchestrator, instead of a local Docker sandbox.

The agents still WRITE harness.py / method.py on the CPU box (Docker); only the
scored RUN is offloaded to a GPU host (e.g. a group DGX card). This preserves the
anti-cheat model unchanged -- the trusted ``_orchestrator.py`` deletes harness.py
and runs the method in an isolated subprocess ON THE GPU NODE -- while the training
uses the GPU. Same return shape as ``runner.run_experiment`` so the pipeline can
route to it transparently.

Transport is plain ``ssh``/``scp`` (the host is reachable, e.g. via a ProxyJump in
``~/.ssh/config``); a single card is pinned with ``CUDA_VISIBLE_DEVICES``. Config via
args or env: ``AUTORESEARCH_GPU_HOST`` / ``AUTORESEARCH_GPU_PYTHON`` /
``AUTORESEARCH_GPU_CUDA_DEVICE`` / ``AUTORESEARCH_GPU_REMOTE_ROOT``.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from autoresearch.experiments.general.design import sanitize_dependencies

_GEN_DIR = Path(__file__).resolve().parent
_TRUSTED_MODULES = ("_orchestrator.py", "isolated_method.py", "_method_driver.py", "contract.py")

_HOST_ENV = "AUTORESEARCH_GPU_HOST"
_PY_ENV = "AUTORESEARCH_GPU_PYTHON"
_DEV_ENV = "AUTORESEARCH_GPU_CUDA_DEVICE"
_ROOT_ENV = "AUTORESEARCH_GPU_REMOTE_ROOT"
_TIMEOUT_ENV = "AUTORESEARCH_GEN_RUN_TIMEOUT"

_DEFAULT_HOST = "gpu-worker-01"
_DEFAULT_PY = "/home/researcher/anaconda3/bin/python"
_DEFAULT_DEV = "7"
# on the training storage (DDN /srv/shared), NOT home (100GB quota) -- cluster rule.
_DEFAULT_ROOT = "/srv/dagloop/ar_gpu_runs"
_DEFAULT_TIMEOUT = 1800.0

_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
_SCP = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-q"]

# A single run fires ~6 ssh/scp ops (mkdir, scp, run, cat, rm...). Through a ProxyJump
# each is a fresh handshake, and that BURST gets refused by the jump/host with these
# signatures (SSH rate-limit / MaxStartups). We (a) multiplex all ops over ONE master
# connection and (b) retry a refused op with backoff -- so the burst becomes 1 handshake.
_CONN_ERR_SIGNS = (
    "kex_exchange_identification", "connection closed by", "connection refused",
    "connection reset", "connection timed out", "could not resolve hostname",
    "port 65535", "broken pipe", "operation timed out",
)


def _is_conn_error(res) -> bool:
    txt = ((getattr(res, "stderr", "") or "") + (getattr(res, "stdout", "") or "")).lower()
    return any(sign in txt for sign in _CONN_ERR_SIGNS)


def _cfg(explicit, env, default):
    return explicit if explicit is not None else os.environ.get(env, default)


def _fail(seed, err):
    return {"valid": False, "method_metric": None, "baseline_metric": None,
            "seed": int(seed), "error": str(err)[:2000], "isolated": False}


def run_experiment_remote(harness_code, method_code, seed, *, host=None, python_bin=None,
                          cuda_device=None, remote_root=None, dependencies=(),
                          io_contract=None, timeout=None, runner=subprocess.run) -> dict:
    """Run one seed on the GPU host through the trusted orchestrator; return the same
    dict shape as ``runner.run_experiment`` (``baseline_metric``/``method_metric``/
    ``valid``/``seed``/``isolated`` + diagnostics). Any transport/parse error fails
    CLOSED (valid=False, isolated=False). ``runner`` is injected for testing."""
    host = _cfg(host, _HOST_ENV, _DEFAULT_HOST)
    python_bin = _cfg(python_bin, _PY_ENV, _DEFAULT_PY)
    cuda_device = str(_cfg(cuda_device, _DEV_ENV, _DEFAULT_DEV))
    root = _cfg(remote_root, _ROOT_ENV, _DEFAULT_ROOT)
    tmo = float(timeout if timeout is not None else os.environ.get(_TIMEOUT_ENV, _DEFAULT_TIMEOUT))

    rdir = None

    # SSH connection multiplexing: every ssh/scp op reuses ONE master connection (opened
    # by the first op), so a run's ~6 ops don't open ~6 fresh handshakes through the jump
    # host -- the burst that gets refused with kex_exchange_identification / "Connection
    # closed by remote host". ControlPersist keeps the master alive between ops.
    cm_dir = tempfile.mkdtemp(prefix="ar-gpu-cm-")
    cpath = os.path.join(cm_dir, "cm.sock")
    _mux = ["-o", "ControlMaster=auto", "-o", f"ControlPath={cpath}", "-o", "ControlPersist=60"]
    ssh_cmd = [*_SSH, *_mux]
    scp_cmd = [*_SCP, *_mux]

    def _attempt(argv, t, retries=3):
        """Run one transport op; retry only a CONNECTION-level refusal (burst/blip) with
        backoff. A clean non-zero exit (e.g. the orchestrator failing) is NOT retried --
        that is a real result, not a transport error."""
        res = None
        for i in range(retries):
            res = runner(argv, capture_output=True, text=True, timeout=t)
            if res.returncode == 0 or not _is_conn_error(res):
                return res
            time.sleep(2 * (i + 1))  # 2s, 4s backoff before retrying a refused handshake
        return res

    def sh(cmd, t):
        return _attempt([*ssh_cmd, host, cmd], t)

    try:
        # Resolve a leading "~" to the remote $HOME: shlex.quote() would quote the
        # tilde so the remote shell wouldn't expand it (mkdir'ing a literal "~" dir
        # while scp's own ~ expands elsewhere). Absolute paths are safe to quote.
        if root.startswith("~"):
            hr = sh("echo $HOME", 30)
            home = (hr.stdout or "").strip()
            if hr.returncode != 0 or not home:
                return _fail(seed, f"could not resolve remote $HOME: {hr.stderr}")
            root = home + root[1:]
        rdir = f"{root}/run_{int(seed)}_{os.getpid()}"
        # 1) assemble the bundle locally: agent files + fixed trusted modules
        with tempfile.TemporaryDirectory(prefix="gpu-run-") as td:
            (Path(td) / "harness.py").write_text(harness_code, encoding="utf-8")
            (Path(td) / "method.py").write_text(method_code, encoding="utf-8")
            (Path(td) / "io_contract.json").write_text(
                json.dumps(io_contract or {}), encoding="utf-8")
            for name in _TRUSTED_MODULES:
                (Path(td) / name).write_text((_GEN_DIR / name).read_text(), encoding="utf-8")
            # 2) remote workdir + scp the whole bundle
            mk = sh(f"rm -rf {shlex.quote(rdir)} && mkdir -p {shlex.quote(rdir)}", 60)
            if mk.returncode != 0:
                return _fail(seed, f"remote mkdir failed: {mk.stderr}")
            files = [str(p) for p in Path(td).iterdir()]
            cp = _attempt([*scp_cmd, *files, f"{host}:{rdir}/"], 180)
            if cp.returncode != 0:
                return _fail(seed, f"scp failed: {cp.stderr}")

        # 3) install any declared deps (best-effort), then run the trusted orchestrator
        #    on the pinned card. deps are sanitized before hitting the remote shell.
        deps = sanitize_dependencies(list(dependencies))
        pip = (f"{shlex.quote(python_bin)} -m pip install -q " + " ".join(map(shlex.quote, deps))
               if deps else "true")
        run = (f"cd {shlex.quote(rdir)} && {pip} && rm -f result.json && "
               f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_device)} "
               f"{shlex.quote(python_bin)} _orchestrator.py {int(seed)}")
        r = sh(run, tmo)
        # isolation is STRUCTURAL: the trusted path completed iff exit 0
        if r.returncode != 0:
            return _fail(seed, r.stderr or r.stdout or "orchestrator exited non-zero")

        # 4) fetch + parse the trusted result
        rr = sh(f"cat {shlex.quote(rdir)}/result.json", 60)
        try:
            data = json.loads(rr.stdout)
        except (json.JSONDecodeError, TypeError):
            return _fail(seed, "unreadable result.json")
        data["isolated"] = True
        return data
    except subprocess.TimeoutExpired:
        return _fail(seed, "remote run timed out")
    except Exception as exc:  # noqa: BLE001 -- transport failures fail closed, never crash the pipeline
        return _fail(seed, f"{type(exc).__name__}: {exc}")
    finally:
        if rdir:
            try:
                runner([*ssh_cmd, host, f"rm -rf {shlex.quote(rdir)}"],
                       capture_output=True, text=True, timeout=30)
            except Exception:  # noqa: BLE001
                pass
        # close the multiplexed master connection, then drop its local socket dir
        try:
            runner([*ssh_cmd, "-O", "exit", host],
                   capture_output=True, text=True, timeout=15)
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(cm_dir, ignore_errors=True)


__all__ = ["run_experiment_remote"]
