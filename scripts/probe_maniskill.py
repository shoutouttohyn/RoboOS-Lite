"""ManiSkill 环境探针：验证安装、观测结构、成功信号。
用法: python scripts/probe_maniskill.py
"""
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (注册环境)

for task_id in ["PickCube-v1"]:
    print(f"=== {task_id} ===", flush=True)
    env = gym.make(task_id, obs_mode="state", render_mode="rgb_array",
                   sim_backend="cpu", render_backend="cpu")
    obs, info = env.reset(seed=0)
    print("obs type:", type(obs).__name__, flush=True)
    if hasattr(obs, "keys"):
        print("obs keys:", list(obs.keys())[:10], flush=True)
    print("info keys:", list(info.keys())[:10] if isinstance(info, dict) else info, flush=True)
    print("action space:", env.action_space, flush=True)

    import numpy as np
    for i in range(3):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"step {i}: rew={rew:.3f} terminated={terminated} "
              f"info_keys={list(info.keys())[:8]}", flush=True)
        if "success" in info:
            print("  success:", info["success"], flush=True)
    env.close()
    print("OK", flush=True)
