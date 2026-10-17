"""逐步定位 segfault：env 创建 → planner 初始化 → solve 各阶段打印。
用法: PYTHONPATH=. python scripts/probe_planner_steps.py
"""
import os

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")

import gymnasium as gym
import mani_skill.envs  # noqa: F401

print("[1] making env...", flush=True)
env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array",
               sim_backend="cpu", render_backend="cpu")
print("[2] env made, resetting...", flush=True)
obs, info = env.reset(seed=1)
print("[3] reset OK", flush=True)

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

print("[4] importing solver OK, constructing planner...", flush=True)
planner = PandaArmMotionPlanningSolver(
    env.unwrapped, debug=False, vis=False,
    base_pose=env.unwrapped.agent.robot.pose,
    print_env_info=False,
)
print("[5] planner constructed OK", flush=True)
planner.close()
env.close()
print("[6] ALL OK", flush=True)
