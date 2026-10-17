"""端到端验证：DeepSeek API + RoboOS Master 规划链路。
运行：cd RoboOS/master && python ../../scripts/test_master_e2e.py
"""
import os
import sys

ROBO_AGENT = "/mnt/d/Kimi code work/robo-agent"
os.chdir(os.path.join(ROBO_AGENT, "RoboOS", "master"))
sys.path.insert(0, ROBO_AGENT)

from robo_agent.integration.enhanced_agent import (  # noqa: E402
    EnhancedGlobalAgent,
    load_master_config,
)

cfg = load_master_config(os.path.join(ROBO_AGENT, "config", "master_deepseek.yaml"))
agent = EnhancedGlobalAgent(cfg)
print("=== Master 初始化成功 ===")

task = "把红色方块放进碗里"
resp = agent.planner.forward(task)
print("=== Planner 原始输出（前 600 字）===")
print(resp[:600])

parsed = agent._extract_json(resp)
print("=== JSON 提取 ===")
print("合法:", agent.reasoning_and_subtasks_is_right(parsed))
