"""
Opaque artifact refs / 不透明产物引用。

A ref is an opaque string handed back by SpecStore.put / put_blob. Callers
treat it as a token; only this module knows the internal shape.
ref 是 SpecStore 返回给调用方的不透明字符串，调用方只当令牌用，
解析格式的逻辑只允许出现在本模块。

Format / 格式: ``spec://{kind}/{session_id}/{key}``
Must satisfy the cross-system regex ``^[a-z]+://.+``.
"""

from __future__ import annotations

import re
from typing import NamedTuple

SCHEME = "spec"

# Cross-system contract: every ref must match this. / 跨系统契约：所有 ref 必须匹配。
REF_RE = re.compile(r"^[a-z]+://.+")


class RefParts(NamedTuple):
    """Decoded components of a ref. / ref 解码后的各组成部分。"""

    kind: str
    session_id: str
    key: str


def encode(kind: str, session_id: str, key: str) -> str:
    """Build the canonical (deterministic) ref. / 构造规范化、确定性的 ref。

    Deterministic over (kind, session_id, key) so puts are idempotent.
    对 (kind, session_id, key) 确定，从而保证 put 幂等。
    """
    return f"{SCHEME}://{kind}/{session_id}/{key}"


def decode(ref: str) -> RefParts:
    """Parse a ref back into its parts. / 把 ref 解析回各组成部分。

    Raises ValueError on a malformed ref. / 格式非法时抛 ValueError。
    Note: key may itself contain '/', so we split with a bounded maxsplit.
    注意：key 自身可能含 '/'，因此用受限的 maxsplit 切分。
    """
    if not REF_RE.match(ref):
        raise ValueError(f"malformed ref: {ref!r}")
    prefix = f"{SCHEME}://"
    if not ref.startswith(prefix):
        raise ValueError(f"unexpected scheme in ref: {ref!r}")
    body = ref[len(prefix):]
    parts = body.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"malformed ref body: {ref!r}")
    kind, session_id, key = parts
    return RefParts(kind=kind, session_id=session_id, key=key)
