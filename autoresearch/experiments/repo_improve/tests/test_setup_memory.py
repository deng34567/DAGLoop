"""Setup 记忆:把层3 摸出来的安装固化,下次同仓库直接复用(先验证再信)。

纪律(与用户/老师讨论定下的三条防反噬):
  - 过期 -> 用前廉价验证,不过就退回阶梯重摸
  - 错误固化 -> 记录必须带证据(实测指标/退出码),不只记命令
  - 过度泛化 -> 记录必须带上下文(卡/预算),换环境不能瞎用
"""
from __future__ import annotations

import json

from autoresearch.experiments.repo_improve.setup_memory import (
    SetupMemory,
    remember_setup,
    recall_setup,
    setup_image_tag,
)


def _mem(tmp_path):
    return SetupMemory(str(tmp_path / "setup_memory.json"))


def test_remember_then_recall_roundtrip(tmp_path):
    m = _mem(tmp_path)
    remember_setup(m, repo_url="https://github.com/karpathy/nanochat",
                   prepare_cmd="uv sync --extra gpu && uv run python -m scripts.tok_train",
                   run_cmd="uv run python -m scripts.base_train --depth=4",
                   metric="Validation bpb", lower_is_better=True,
                   metric_value=2.831459, budget_s=300, gpu="NVIDIA L20")

    hit = recall_setup(m, repo_url="https://github.com/karpathy/nanochat",
                       budget_s=300, gpu="NVIDIA L20")
    assert hit is not None
    assert hit["run_cmd"].endswith("--depth=4")
    assert hit["metric"] == "Validation bpb"
    # 证据:必须记下当时实测到的指标值,否则无从判断这条记录是否可信
    assert hit["evidence"]["metric_value"] == 2.831459
    assert hit["recorded_at"]                      # 时间戳,用于判断陈旧


def test_recall_requires_matching_context(tmp_path):
    # 过度泛化防线:同仓库但换了卡/预算,之前调的微配置未必成立 -> 不命中
    m = _mem(tmp_path)
    remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd="c", metric="m",
                   lower_is_better=True, metric_value=1.0, budget_s=300, gpu="NVIDIA L20")

    assert recall_setup(m, repo_url="r", budget_s=300, gpu="NVIDIA L20") is not None
    assert recall_setup(m, repo_url="r", budget_s=600, gpu="NVIDIA L20") is None   # 预算变了
    assert recall_setup(m, repo_url="r", budget_s=300, gpu="NVIDIA H100") is None  # 卡变了
    assert recall_setup(m, repo_url="other", budget_s=300, gpu="NVIDIA L20") is None


def test_unverified_setups_are_never_remembered(tmp_path):
    # 错误固化防线:没经硬判的东西一律不入库
    m = _mem(tmp_path)
    remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd="c", metric="m",
                   lower_is_better=True, metric_value=None, budget_s=300, gpu="L20")
    assert recall_setup(m, repo_url="r", budget_s=300, gpu="L20") is None


def test_remembering_the_same_repo_again_overwrites(tmp_path):
    m = _mem(tmp_path)
    for cmd, val in (("old", 3.0), ("new", 2.5)):
        remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd=cmd, metric="m",
                       lower_is_better=True, metric_value=val, budget_s=300, gpu="L20")
    hit = recall_setup(m, repo_url="r", budget_s=300, gpu="L20")
    assert hit["run_cmd"] == "new" and hit["evidence"]["metric_value"] == 2.5
    assert len(json.loads(open(m.path).read())) == 1     # 不重复堆积


def test_image_tag_is_deterministic_per_context(tmp_path):
    a = setup_image_tag("https://github.com/x/y", 300, "3")
    assert a == setup_image_tag("https://github.com/x/y", 300, "3")     # 同上下文 -> 同名(可覆盖)
    assert a.startswith("autoresearch-setup-") and a.islower()          # docker tag 合法
    assert a != setup_image_tag("https://github.com/x/y", 600, "3")     # 换预算 -> 换名
    assert a != setup_image_tag("https://github.com/x/y", 300, None)    # 换卡 -> 换名


def test_frozen_image_name_round_trips(tmp_path):
    m = _mem(tmp_path)
    remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd="c", metric="m",
                   lower_is_better=True, metric_value=2.0, budget_s=300, gpu="L20",
                   image="autoresearch-setup-abc123")
    hit = recall_setup(m, repo_url="r", budget_s=300, gpu="L20")
    assert hit["image"] == "autoresearch-setup-abc123"
    # 没冻结镜像时字段为 None(下次 recall 会重新发现,而不是拿裸镜像瞎跑)
    remember_setup(m, repo_url="r2", prepare_cmd="p", run_cmd="c", metric="m",
                   lower_is_better=True, metric_value=2.0, budget_s=300, gpu="L20")
    assert recall_setup(m, repo_url="r2", budget_s=300, gpu="L20")["image"] is None


def test_corrupt_store_degrades_to_empty(tmp_path):
    # 记忆坏了不能把 pipeline 带崩,退化成"没有记忆"即可
    p = tmp_path / "setup_memory.json"
    p.write_text("{{{ not json")
    m = SetupMemory(str(p))
    assert recall_setup(m, repo_url="r", budget_s=300, gpu="L20") is None
    remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd="c", metric="m",
                   lower_is_better=True, metric_value=1.0, budget_s=300, gpu="L20")
    assert recall_setup(m, repo_url="r", budget_s=300, gpu="L20") is not None


# --- #41: absolute session-workspace paths must be portablized so a setup remembered in one
#     session/deployment's workspace runs in another's (cross-deployment ~/.cache pollution). ---
CIFAR_ABS = ('cd /data/researcher/ar-web-public/repo_improve_local/'
             'ce9f00bb-85e9-4b87-9501-c371b7ab677e/_sandbox/repo\n'
             '/usr/local/bin/python -c "import airbench; airbench.train94()"')
MINGPT_ABS = ('REPO=/data/researcher/ar-web-public/repo_improve_local/'
              'f3f3781e-9b28-45ca-aabd-6cf1e9492305/_sandbox/repo && '
              'cd "$REPO/projects/chargpt" && "$REPO/.venv/bin/python" chargpt.py')


def test_remember_portablizes_absolute_workspace_cd(tmp_path):
    m = SetupMemory(str(tmp_path / "s.json"))
    remember_setup(m, repo_url="https://github.com/KellerJordan/cifar10-airbench",
                   prepare_cmd="set -e", run_cmd=CIFAR_ABS, metric="acc", lower_is_better=False,
                   metric_value=0.94, budget_s=600, gpu="4")
    hit = recall_setup(m, repo_url="https://github.com/KellerJordan/cifar10-airbench",
                       budget_s=600, gpu="4")
    assert "ar-web-public" not in hit["run_cmd"]          # the stale abs workspace path is gone
    assert "/data/researcher/ar-web-public" not in hit["run_cmd"]
    assert "$PWD" in hit["run_cmd"]                        # rewritten to the current repo dir


def test_variable_form_workspace_path_portablized(tmp_path):
    m = SetupMemory(str(tmp_path / "s.json"))
    remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd=MINGPT_ABS, metric="loss",
                   lower_is_better=True, metric_value=1.2, budget_s=1800, gpu="3")
    hit = recall_setup(m, repo_url="r", budget_s=1800, gpu="3")
    assert "f3f3781e" not in hit["run_cmd"]
    assert 'REPO="$PWD"' in hit["run_cmd"]                 # REPO now points at the current repo dir
    assert '"$REPO/.venv/bin/python"' in hit["run_cmd"]    # the rest (relative to $REPO) is preserved


def test_recall_cleans_legacy_polluted_entries(tmp_path):
    # an entry written BEFORE the fix (raw absolute path on disk) is cleaned on read, so an old
    # ~/.cache doesn't keep crashing every recall.
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"r|600|4": {"repo_url": "r", "prepare_cmd": "set -e",
                 "run_cmd": CIFAR_ABS, "metric": "acc", "lower_is_better": False,
                 "context": {"budget_s": 600, "gpu": "4"}, "evidence": {"metric_value": 0.94}}}),
                 encoding="utf-8")
    hit = recall_setup(SetupMemory(str(p)), repo_url="r", budget_s=600, gpu="4")
    assert "ar-web-public" not in hit["run_cmd"] and "$PWD" in hit["run_cmd"]


def test_normal_commands_are_untouched(tmp_path):
    m = SetupMemory(str(tmp_path / "s.json"))
    cmd = 'cd projects/chargpt && python chargpt.py --max_iters=100'   # relative subdir cd -> keep
    remember_setup(m, repo_url="r", prepare_cmd="p", run_cmd=cmd, metric="m",
                   lower_is_better=True, metric_value=1.0, budget_s=300, gpu="L20")
    assert recall_setup(m, repo_url="r", budget_s=300, gpu="L20")["run_cmd"] == cmd
