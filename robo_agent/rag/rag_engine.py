"""RAG 引擎：LlamaIndex + Qdrant，双知识库。

库 A「skill_docs」：技能文档（名称/参数/前置条件/失败模式），规划阶段检索，
                  让 Planner 知道"有哪些工具可调用、什么时候调用"。
库 B「domain_kb」：领域知识库（工业/家庭操作手册语料）。

Phase 1: 纯向量检索 Top-K 注入 prompt。
Phase 2:  + BM25 混合检索 + bge-reranker 重排（本文件已留 hook）。
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ..memory.qdrant_client_factory import make_qdrant_client


class KnowledgeBase:
    """单个知识库 = 一个 Qdrant collection + 一个 LlamaIndex index。"""

    def __init__(self, name: str, qdrant_url: str | None = None,
                 embed_model: str = "BAAI/bge-m3"):
        self.name = name
        # llama-index 默认 cache 目录与 HF hub 缓存不一致会离线加载失败，
        # 显式指向 HF hub 缓存根目录（含 models--* 的那层）
        import os
        hf_hub = os.path.join(
            os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")),
            "hub")
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model,
                                                cache_folder=hf_hub)
        self.client = make_qdrant_client(qdrant_url)
        self.store = QdrantVectorStore(client=self.client, collection_name=name)
        self.ctx = StorageContext.from_defaults(vector_store=self.store)
        self.index: VectorStoreIndex | None = None
        self.reranker = None  # Phase 2 通过 enable_rerank() 开启（bge-reranker）

    def build_from_dir(self, docs_dir: str | Path) -> int:
        """从目录加载文档（支持 md/txt/pdf），建索引，返回文档数。"""
        docs = SimpleDirectoryReader(str(docs_dir)).load_data()
        self.index = VectorStoreIndex.from_documents(
            docs, storage_context=self.ctx, embed_model=self.embed_model)
        return len(docs)

    def build_from_texts(self, texts: list[str]) -> None:
        self.index = VectorStoreIndex.from_documents(
            [Document(text=t) for t in texts], storage_context=self.ctx,
            embed_model=self.embed_model)

    def enable_rerank(self, model: str = "BAAI/bge-reranker-base") -> None:
        """Phase 2：开启重排。需在 query 时配合 similarity_top_k 放大召回。"""
        from llama_index.postprocessor.flag_embedding_reranker import (
            FlagEmbeddingReranker,
        )
        self.reranker = FlagEmbeddingReranker(model=model, top_n=5)

    def query(self, text: str, top_k: int = 5) -> str:
        """检索并格式化为可注入 prompt 的字符串。索引未建时返回空串。"""
        if self.index is None and not self._load_existing():
            return ""
        retriever = self.index.as_retriever(
            similarity_top_k=top_k * 4 if self.reranker else top_k)
        nodes = retriever.retrieve(text)
        if self.reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=text)
        if not nodes:
            return ""
        lines = [f"[知识检索:{self.name}]"]
        for i, n in enumerate(nodes, 1):
            lines.append(f"{i}. {n.get_content().strip()[:500]}")
        return "\n".join(lines)

    def _load_existing(self) -> bool:
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.store, embed_model=self.embed_model)
            return True
        except Exception:
            return False


class RAGEngine:
    """ Planner 的检索入口：一次查询，两个库都查。"""

    def __init__(self, qdrant_url: str | None = None, kb_root: str | Path = "knowledge"):
        self.kb_root = Path(kb_root)
        self.skill_kb = KnowledgeBase("skill_docs", qdrant_url)
        self.domain_kb = KnowledgeBase("domain_kb", qdrant_url)

    def build_skill_docs(self, skill_md_dir: str | Path) -> int:
        """技能文档库：每个技能一篇 markdown，见 knowledge/skill_docs/ 示例。"""
        return self.skill_kb.build_from_dir(skill_md_dir)

    def build_domain_kb(self, docs_dir: str | Path) -> int:
        return self.domain_kb.build_from_dir(docs_dir)

    def context_for(self, task_text: str) -> str:
        """规划前调用：拼接技能知识 + 领域知识的检索结果。"""
        parts = [self.skill_kb.query(task_text), self.domain_kb.query(task_text)]
        return "\n\n".join(p for p in parts if p)


if __name__ == "__main__":
    # 冒烟测试前：先在 WSL 里启动 Qdrant，并准备 knowledge/ 下的文档
    rag = RAGEngine()
    rag.build_skill_docs("knowledge/skill_docs")
    print(rag.context_for("把红色方块放进碗里"))
