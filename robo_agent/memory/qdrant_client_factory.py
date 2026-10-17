"""Qdrant 客户端工厂：优先连服务器，失败/指定时退回本地嵌入式模式。

用法：
    client = make_qdrant_client()                      # 读 QDRANT_URL 环境变量
    client = make_qdrant_client("local:./qdrant_data") # 显式本地模式

注意：本地模式的数据目录必须在 WSL 原生文件系统上（如 ~/...），
不要放 /mnt/d（DrvFS 的文件锁不可靠，会产生幽灵锁）。
"""
from __future__ import annotations

import os

from qdrant_client import QdrantClient

DATA_ROOT = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "robo-agent", "qdrant")


def make_qdrant_client(url: str | None = None) -> QdrantClient:
    url = url or os.environ.get("QDRANT_URL") or "local"
    if url.startswith("local"):
        # local 或 local:path → 嵌入式模式（无需 Docker，数据落盘）
        path = url.split(":", 1)[1] if ":" in url else DATA_ROOT
        os.makedirs(path, exist_ok=True)
        # Qdrant 本地模式不允许同一存储目录开多个 client，按路径复用单例
        real = os.path.realpath(path)
        if real not in _CLIENTS:
            _CLIENTS[real] = QdrantClient(path=path)
        return _CLIENTS[real]
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return QdrantClient(url=url)


_CLIENTS: dict[str, QdrantClient] = {}
