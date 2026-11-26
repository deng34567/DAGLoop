# DAGLoop / AutoResearch

This repository contains the implementation for **DAGLoop: Aligning Method Design with
Experimental Feedback in Automated AI Research**. DAGLoop designs a Method, decomposes it into a
dependency DAG of components, executes repository-owned evaluations, feeds verification evidence
back into Method revision and Memory, and produces an evidence-bound final report.

## Repository layout

| Path | Contents |
|---|---|
| `autoresearch/orchestrator/` | Research loop, state transitions, and session management |
| `autoresearch/agents/` | Research agents, prompts, and shared execution helpers |
| `autoresearch/experiments/` | Repository execution, evaluation, and checkpoint handling |
| `autoresearch/web/` | Local controller and viewer |
| `autoresearch/**/tests/` | Unit and workflow regression tests |
| `scripts/dgx/` | Optional SSH deployment entry point and its tests |
| `scripts/smoke_*.py` | Explicitly invoked smoke checks |
| `.env.example` | Endpoint, model, device, and work-directory configuration template |

## Requirements

- Linux host (the documented Docker setup has been validated on Linux)
- Python 3.13
- Git
- A running Docker daemon with permission to start containers
- `uv` for the installation and launch commands below
- An OpenAI-compatible Chat Completions endpoint, an API key, and a model with tool calling
- For GPU tasks: NVIDIA GPUs, host driver libraries, and the device permissions required by the
  target repository

CPU-only tasks do not require a GPU. Task-specific compilers, datasets, weights, and system tools
are defined by the submitted repository rather than bundled into this Python package.

For GPU runs, explicitly set `AUTORESEARCH_LOCAL_GPU` to the device IDs allocated to the
controller (for example, `0` on a machine with one GPU). Remote SSH runners require separate
host, Python, and storage configuration; they are not part of this local Docker quickstart.
Their bundled host and path defaults are examples. Set the SSH host, Python executable,
working directory, and allocated GPU IDs explicitly before using remote execution.

## Install from source

```bash
git clone https://github.com/deng34567/DAGLoop.git
cd DAGLoop
cp .env.example .env
uv sync --locked --extra sandbox
```

The `sandbox` extra installs the Docker/OpenHands execution dependencies used for real repository
experiments. For development, install and retain both extras:

```bash
uv sync --locked --extra sandbox --extra test
uv run --locked --extra sandbox --extra test pytest -q autoresearch scripts/dgx/tests
```

The default test suite mocks model calls and sandbox execution; it does not require an API key
or a GPU. The separate `scripts/smoke_openhands_backend.py` script is opt-in and makes real model
calls when explicitly enabled.

Viewer JavaScript regression tests also use Node.js when available and are skipped otherwise.

## Configure and run

Edit `.env` and set `OPENROUTER_KEY`. The example uses OpenRouter. For another compatible endpoint,
also set `OPENROUTER_BASE_URL`, `KIMI_MODEL`, and `AUTORESEARCH_SANDBOX_MODEL` to that endpoint's
URL and available model identifiers. `KIMI_MODEL` selects the orchestration model;
`AUTORESEARCH_SANDBOX_MODEL` selects the coding model and does not inherit `KIMI_MODEL`.
The sandbox inherits `OPENROUTER_BASE_URL` unless `AUTORESEARCH_SANDBOX_BASE_URL` is set.

Reasoning controls and output limits depend on the endpoint. The example sets sandbox reasoning
to `none`; use `AUTORESEARCH_SANDBOX_REASONING_EFFORT=default` to omit that option when the
endpoint does not accept it. Orchestration reasoning is controlled separately by
`VIBE_REASONING_EFFORT` and is omitted when unset.

For a DeepSeek Flash 0731 endpoint that supports `max` reasoning and a 262144-token completion
limit, the corresponding settings are below. Use the exact model identifiers exposed by your
endpoint and its supported token limit. The direct/Web and sandbox paths inherit
`AUTORESEARCH_LLM_MAX_TOKENS`; `AUTORESEARCH_WEB_LLM_MAX_TOKENS` and
`AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS` can override the paths independently.

```dotenv
KIMI_MODEL=deepseek-v4-flash-0731
AUTORESEARCH_SANDBOX_MODEL=deepseek-v4-flash-0731
VIBE_REASONING_EFFORT=max
AUTORESEARCH_SANDBOX_REASONING_EFFORT=max
AUTORESEARCH_LLM_MAX_TOKENS=262144
```

For real repository experiments, keep these switches enabled:

```dotenv
AUTORESEARCH_GEN_EXPERIMENT=1
AUTORESEARCH_AGENTIC_IMPROVE=1
AUTORESEARCH_REAL_SEARCH=1
AUTORESEARCH_REPO_MODE=auto
AUTORESEARCH_REPO_SPLIT=0
```

`auto` selects CPU or GPU according to the task. `AUTORESEARCH_REPO_SPLIT=0` keeps execution
on the local Docker host. Set the GPU IDs explicitly for GPU tasks; for example,
`AUTORESEARCH_LOCAL_GPU=0` on a machine with one allocated GPU. Enabling split execution
requires a separately configured SSH deployment.

`AUTORESEARCH_REAL_SEARCH=1` enables literature and repository search through academic indexes.
Setting `TAVILY_API_KEY` additionally enables Tavily, with the academic indexes as fallback.

Start the local controller:

```bash
uv run --locked --extra sandbox --env-file .env python -m autoresearch.web.server
```

Then open <http://127.0.0.1:8300>, enter a task with its repository URL, and start the session.
Answer clarification and confirmation prompts as they appear. Runtime state, checkpoints, reports, and the session
database are stored under `AUTORESEARCH_WEB_WORKDIR` (default: `/tmp/autoresearch-web`). Set it to a
persistent directory for real work; `.env.example` uses `./workspace/web`.

The controller is intended for trusted local use. Keep the loopback binding shown above;
the optional login system does not provide isolation between different users' research sessions.

## Downloads, caches, and authorization

The Python package does not include third-party repositories, benchmark datasets, model weights,
or the sandbox container image. The default sandbox image is
`ghcr.io/openhands/agent-server:1.19.1-python`; a first real run may therefore require a substantial
container-image download. Repository-defined setup may also require packages, datasets, or model
artifacts.

Prepare the configured sandbox image on the Docker host before starting the first session.
Dependency installation and Docker's image pull are infrastructure setup steps, separate from
the session's download authorization.

Check the default image with `docker image inspect ghcr.io/openhands/agent-server:1.19.1-python`.
If it is absent, obtain it with `docker pull ghcr.io/openhands/agent-server:1.19.1-python` when
you are ready for that download. Use your configured image name if overriding the default.

The Web start form includes an initial download-authorization checkbox. When selected, that
authorization covers required artifacts from repository-declared or official sources, uses caches
first, and does not authorize unverified sources. If it is not selected, the workflow instructs
the agent to report missing sizeable downloads as blockers pending approval. This checkbox records
the workflow policy; it does not enforce a container network restriction. Private repositories,
gated models, and authenticated registries still require the user to configure the corresponding
credentials; DAGLoop does not bypass access controls.

Repository-experiment package and Hugging Face caches are session-local by default; the
setup-recovery cache is per-user. Package installs use official indexes unless a mirror is
explicitly configured. Optional deployment settings are:

- `AUTORESEARCH_CLONE_MIRROR_DIR`: pre-seeded Git mirrors
- `AUTORESEARCH_SANDBOX_PKG_CACHE_DIR`: shared pip/uv package cache
- `AUTORESEARCH_SANDBOX_HF_HOME`: shared Hugging Face cache
- `AUTORESEARCH_SANDBOX_DATA_ROOT`: read-only or reusable task data root
- `AUTORESEARCH_SANDBOX_CACHE`: setup-recovery cache (defaults to `~/.cache/autoresearch/sandbox`)
- `AUTORESEARCH_SANDBOX_PIP_MIRROR=1`: opt in to the deployment-specific package mirror
- `AUTORESEARCH_SANDBOX_IMAGE`: override the sandbox image

Caches may reduce downloads, but cached optimized implementations, scores, trained outputs, or
artifacts from another research session must not be used as new experimental evidence.

## Result semantics

`Done`, “a real scalar was measured,” and “the complete Method passed final verification” are
different claims. Reports and exported artifacts must identify the selected immutable checkpoint,
its measured result, the evaluator/gates that ran, and any limitations. A run may legitimately end
with a negative result when no candidate qualifies.

Resuming a session preserves its recorded attempts and results.

## Paper

[DAGLoop on arXiv](https://arxiv.org/abs/2610.25089)

## License

The DAGLoop implementation is available under the [MIT License](LICENSE). Third-party
repositories, datasets, models, and bundled assets retain their respective licenses and notices.

