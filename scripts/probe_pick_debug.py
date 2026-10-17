"""抓取过程探针：打印各阶段 TCP/方块/夹爪状态，定位抓取失败原因。
用法: PYTHONUNBUFFERED=1 python scripts/probe_pick_debug.py
"""
import os

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array",
               control_mode="pd_ee_delta_pose",
               sim_backend="cpu", render_backend="cpu").unwrapped
obs, info = env.reset(seed=7)

cube_p = env.cube.pose.p[0].cpu().numpy()
goal_p = env.goal_site.pose.p[0].cpu().numpy()
print("cube:", np.round(cube_p, 3), "goal:", np.round(goal_p, 3), flush=True)
print("tcp init:", np.round(env.agent.tcp.pose.p[0].cpu().numpy(), 3), flush=True)
print("action space:", env.action_space, flush=True)
print("controller:", [c.__class__.__name__ for c in env.agent.controller.configs], flush=True)


def tcp():
    return env.agent.tcp.pose.p[0].cpu().numpy()


def move(target, gripper, max_steps=150, tol=0.015, label=""):
    for i in range(max_steps):
        delta = target - tcp()
        if np.linalg.norm(delta) < tol:
            print(f"  [{label}] reached in {i} steps, tcp={np.round(tcp(),3)}", flush=True)
            return True
        a = np.zeros(7, dtype="float32")
        a[:3] = np.clip(delta * 5.0, -1, 1)
        a[6] = gripper
        env.step(a)
    print(f"  [{label}] NOT reached, tcp={np.round(tcp(),3)} target={np.round(target,3)}", flush=True)
    return False


move(cube_p + [0, 0, 0.15], -1.0, label="hover")
move(cube_p + [0, 0, 0.005], -1.0, label="descend")

print("gripper joints before close:", env.agent.robot.get_qpos().cpu().numpy()[-2:], flush=True)
for sign, name in [(1.0, "+1"), (-1.0, "-1")]:
    env.reset(seed=7)
    move(cube_p + [0, 0, 0.005], -1.0, label=f"descend({name})")
    for _ in range(30):
        a = np.zeros(7, dtype="float32"); a[6] = sign
        env.step(a)
    q = env.agent.robot.get_qpos().cpu().numpy()[-2:]
    _, _, _, _, inf = env.step(np.zeros(7, dtype="float32"))
    print(f"  gripper cmd {name}: joints={np.round(q,3)} is_grasped={inf.get('is_grasped')}", flush=True)
