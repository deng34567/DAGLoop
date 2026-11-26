"""
AutoResearch Spec Store.

Shared artifact store all agents use to exchange products. Write an artifact,
get an opaque ref, pass the ref via events; downstream reads by ref.
各 agent 共享的产物存储：写入拿 ref，经事件传递 ref，下游凭 ref 读取。

Backed by SQLite (structured JSON) + local files (blobs). Synchronous.
底层 SQLite（结构化 JSON）+ 本地文件（blob），同步接口。
"""

from .store import SpecStore, EmitterNotAllowed

__all__ = ["SpecStore", "EmitterNotAllowed"]
