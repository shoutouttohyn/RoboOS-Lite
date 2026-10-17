"""消融评测主运行器：5 任务 × baseline/+RAG/+记忆，逐任务真实仿真执行。

前提：scripts/run_slaver.py（ManiSkill 执行器版）已在运行。
用法：
    python scripts/run_benchmark.py --episodes 3
    python scripts/run_benchmark.py --episodes 3 --tasks t1 t3   # 子集
输出：
    run/results/<config>.json  +  run/results/report.md
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

from robo_agent.eval.benchmark import TASK_SUITE, RESULTS_DIR  # noqa: E402
from robo_agent.eval.report import report  # noqa: E402
from robo_agent.integration.enhanced_agent import (  # noqa: E402
    EnhancedGlobalAgent,
    load_master_config,
)
from robo_agent.memory.memory_manager import COLLECTION, MemoryManager  # noqa: E402
from robo_agent.rag.rag_engine import RAGEngine  # noqa: E402

CONFIGS = [("baseline", {"rag": False, "memory": False}),
           ("rag", {"rag": True, "memory": False}),
           ("full", {"rag": True, "memory": True})]


def wipe_memory(memory: MemoryManager) -> None:
    """清空情景记忆（换配置组前调用，保证公平对比）。"""
    memory.client.delete_collection(COLLECTION)
    memory._init_collection()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--tasks", nargs="*", default=None)
    args = p.parse_args()

    tasks = [t for t in TASK_SUITE if args.tasks is None or t["id"] in args.tasks]
    print(f"任务集: {[t['id'] for t in tasks]}, 每任务 {args.episodes} 局", flush=True)

    rag = RAGEngine(kb_root=ROBO_AGENT / "knowledge")
    n1 = rag.build_skill_docs(ROBO_AGENT / "knowledge" / "skill_docs")
    n2 = rag.build_domain_kb(ROBO_AGENT / "knowledge" / "domain_kb")
    print(f"[rag] skill_docs={n1} domain_kb={n2}", flush=True)
    memory = MemoryManager()

    cfg = load_master_config(str(ROBO_AGENT / "config" / "master_deepseek.yaml"))
    agent = EnhancedGlobalAgent(cfg, rag=rag, memory=memory)
    names = agent.wait_for_robots()
    print("机器人已注册:", names, flush=True)
    assert names, "Slaver 未注册，请先启动 scripts/run_slaver.py"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config_name, flags in CONFIGS:
        wipe_memory(memory)
        rows = []
        for task in tasks:
            for ep in range(args.episodes):
                t0 = time.time()
                try:
                    r = agent.run_task_sync(task["instruction"], config_flags=flags)
                except Exception as e:
                    r = {"task": task["instruction"], "success": False,
                         "failure_reason": f"exception: {e}", "subtasks": [], "n_steps": 0}
                rows.append({
                    "task_id": task["id"], "episode": ep,
                    "success": bool(r.get("success")),
                    "failure_reason": r.get("failure_reason", ""),
                    "n_steps": r.get("n_steps", 0),
                    "latency_s": round(time.time() - t0, 2),
                    "config": config_name,
                })
                print(f"[{config_name}] {task['id']} ep{ep}: "
                      f"success={r.get('success')} ({time.time()-t0:.1f}s)", flush=True)
        # 聚合
        n = len(rows)
        agg = {
            "config": config_name, "n_episodes": n,
            "success_rate": sum(r["success"] for r in rows) / n,
            "avg_steps": sum(r["n_steps"] for r in rows) / n,
            "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
            "episodes": rows,
        }
        (RESULTS_DIR / f"{config_name}.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=2))
        print(f"== {config_name}: success_rate={agg['success_rate']:.1%} ==", flush=True)

    print("\n" + report(), flush=True)


if __name__ == "__main__":
    main()
