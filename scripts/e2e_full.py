"""完整闭环 e2e：Master(含记忆模块) + Redis + Slaver(demo MCP 机器人)。

前提：scripts/run_slaver.py 已在另一个进程运行并完成注册。
运行：python scripts/e2e_full.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROBO_AGENT = Path(__file__).resolve().parent.parent
os.chdir(ROBO_AGENT / "RoboOS" / "master")  # 场景/日志相对路径
sys.path.insert(0, str(ROBO_AGENT))

from robo_agent.integration.enhanced_agent import (  # noqa: E402
    EnhancedGlobalAgent,
    load_master_config,
)
from robo_agent.memory.memory_manager import Episode, MemoryManager  # noqa: E402

memory = MemoryManager()
cfg = load_master_config(str(ROBO_AGENT / "config" / "master_deepseek.yaml"))
agent = EnhancedGlobalAgent(cfg, memory=memory)
print("=== Master 就绪，等待 Slaver 注册 ===", flush=True)

for _ in range(30):
    robots = agent.collaborator.read_all_agents_info()
    if robots:
        break
    time.sleep(1)
print("已注册机器人:", list(robots), flush=True)
assert robots, "Slaver 未注册，请先启动 scripts/run_slaver.py"

# 先注入两条历史经验，验证记忆召回
memory.write(Episode(
    task="grasp the apple from the kitchen table",
    subtasks=["navigate to kitchen table", "grasp the apple"],
    success=False, failure_reason="target object was occluded by the knife"))
memory.write(Episode(
    task="place the apple into the bowl",
    subtasks=["navigate to serving table", "place apple into bowl"],
    success=True))

task = ("Navigate to the kitchen table, grasp the apple, "
        "then place it into the bowl on the serving table.")
print(f"=== 下发任务: {task} ===", flush=True)

result = agent.run_task_sync(task, config_flags={"rag": False, "memory": True})
print("=== 任务结果 ===")
print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
