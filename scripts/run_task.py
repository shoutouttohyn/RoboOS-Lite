"""通用任务运行器：单任务 e2e，支持 RAG / 记忆开关（消融实验的 executor）。

用法：
    python scripts/run_task.py --task "把红色方块放进碗里" --rag --memory
    python scripts/run_task.py --task "..."            # baseline
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROBO_AGENT = Path(__file__).resolve().parent.parent
os.chdir(ROBO_AGENT / "RoboOS" / "master")
sys.path.insert(0, str(ROBO_AGENT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROBO_AGENT / ".env")

from robo_agent.integration.enhanced_agent import (  # noqa: E402
    EnhancedGlobalAgent,
    load_master_config,
)
from robo_agent.memory.memory_manager import MemoryManager  # noqa: E402
from robo_agent.rag.rag_engine import RAGEngine  # noqa: E402


def build_rag() -> RAGEngine:
    rag = RAGEngine(kb_root=ROBO_AGENT / "knowledge")
    n_skills = rag.build_skill_docs(ROBO_AGENT / "knowledge" / "skill_docs")
    n_domain = rag.build_domain_kb(ROBO_AGENT / "knowledge" / "domain_kb")
    print(f"[rag] skill_docs={n_skills}, domain_kb={n_domain}", flush=True)
    return rag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--rag", action="store_true")
    p.add_argument("--memory", action="store_true")
    p.add_argument("--episodes-context", type=int, default=3)
    args = p.parse_args()

    memory = MemoryManager() if args.memory else None
    rag = build_rag() if args.rag else None

    cfg = load_master_config(str(ROBO_AGENT / "config" / "master_deepseek.yaml"))
    agent = EnhancedGlobalAgent(cfg, rag=rag, memory=memory)

    for _ in range(30):
        robots = agent.collaborator.read_all_agents_info()
        if robots:
            break
        time.sleep(1)
    if not robots:
        print(json.dumps({"error": "no robot registered"}))
        sys.exit(1)

    t0 = time.time()
    result = agent.run_task_sync(
        args.task, config_flags={"rag": args.rag, "memory": args.memory})
    result["latency_s"] = round(time.time() - t0, 2)
    result["config"] = {"rag": args.rag, "memory": args.memory}
    print("===RESULT===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
