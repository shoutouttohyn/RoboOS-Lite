"""长期记忆模块：Qdrant 向量库 + bge-m3 embedding。

两条核心路径：
- recall(task_text): 规划前召回 Top-K 条相似历史经验，注入 Planner prompt。
- write(episode):   任务结束后写入情景记忆 (任务, 子任务序列, 成败, 失败原因)。

短期记忆不在本模块内 —— 由 RoboOS 原生 Redis Shared Memory 承担。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .qdrant_client_factory import make_qdrant_client

COLLECTION = "episodic_memory"
EMBED_DIM = 1024  # bge-m3


@dataclass
class Episode:
    """一条情景记忆 = 一次任务执行的总结。"""
    task: str                              # 自然语言任务描述
    subtasks: list[str] = field(default_factory=list)
    success: bool = True
    failure_reason: str = ""
    tags: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_text(self) -> str:
        status = "成功" if self.success else f"失败(原因: {self.failure_reason})"
        plan = " -> ".join(self.subtasks) if self.subtasks else "(无子任务)"
        return f"任务: {self.task} | 执行计划: {plan} | 结果: {status}"

    def to_payload(self) -> dict:
        return {
            "task": self.task,
            "subtasks": self.subtasks,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "tags": self.tags,
            "ts": self.ts,
        }


class MemoryManager:
    """长期记忆管理器。Embedding 默认 CPU 跑 bge-m3，无需 GPU。"""

    def __init__(self, qdrant_url: str | None = None, embed_model: str = "BAAI/bge-m3"):
        self.client = make_qdrant_client(qdrant_url)
        self._init_collection()
        # 用 sentence-transformers 而非 FlagEmbedding：
        # FlagEmbedding 1.4.x 与 transformers 5.x 的 tokenizer.pad 存在兼容 bug
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model)

    def _init_collection(self) -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if COLLECTION not in names:
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode([text], normalize_embeddings=True)[0].tolist()
    def recall(self, task_text: str, top_k: int = 3) -> str:
        """规划前召回：返回可直接注入 prompt 的经验字符串。无历史时返回空串。"""
        hits = self.client.query_points(
            COLLECTION, query=self._embed(task_text), limit=top_k,
        ).points
        if not hits:
            return ""
        lines = ["[历史经验召回] 以下相似任务的执行记录供参考："]
        for i, h in enumerate(hits, 1):
            ep = Episode(**{k: v for k, v in h.payload.items() if k in Episode.__dataclass_fields__})
            lines.append(f"{i}. {ep.to_text()}")
        return "\n".join(lines)

    def write(self, episode: Episode) -> str:
        """任务结束后写入一条情景记忆，返回记忆 ID。"""
        point_id = str(uuid.uuid4())
        self.client.upsert(COLLECTION, points=[PointStruct(
            id=point_id,
            vector=self._embed(episode.to_text()),
            payload=episode.to_payload(),
        )])
        return point_id


if __name__ == "__main__":
    # 冒烟测试: python -m memory.memory_manager
    mm = MemoryManager()
    mm.write(Episode(task="把红色方块放到碗里", subtasks=["定位红方块", "抓取", "移动到碗上方", "放置"], success=True))
    mm.write(Episode(task="抓取桌上的杯子", subtasks=["定位杯子", "抓取"], success=False, failure_reason="抓取点偏移，杯子滑落"))
    print(mm.recall("把杯子放进碗"))
