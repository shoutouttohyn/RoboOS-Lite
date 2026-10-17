"""推方块探针：打印推前/推后的方块与目标位置、距离。
用法: PYTHONUNBUFFERED=1 python scripts/probe_push_debug.py
"""
import os

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

env = gym.make("PushCube-v1", obs_mode="state", render_mode="rgb_array",
               control_mode="pd_ee_delta_pose",
               sim_backend="cpu", render_backend="cpu").unwrapped
env.reset(seed=3)
OPEN, CLOSE = 1.0, -1.0


def tcp():
    return env.agent.tcp.pose.p[0].cpu().numpy()


def move(target, gripper, max_steps=400, tol=0.02, label=""):
    for i in range(max_steps):
        delta = target - tcp()
        if np.linalg.norm(delta) < tol:
            print(f"  [{label}] reached@{i} tcp={np.round(tcp(),3)}", flush=True)
            return True
        a = np.zeros(7, dtype="float32")
        a[:3] = np.clip(delta * 5.0, -1, 1)
        a[6] = gripper
        env.step(a)
    print(f"  [{label}] stopped tcp={np.round(tcp(),3)} tgt={np.round(target,3)}", flush=True)
    return False


cube_p = env.obj.pose.p[0].cpu().numpy()
goal_p = env.goal_region.pose.p[0].cpu().numpy()
print("cube:", np.round(cube_p, 3), "goal:", np.round(goal_p, 3), flush=True)

d = goal_p[:2] - cube_p[:2]
dist = float(np.linalg.norm(d))
dirv = np.concatenate([d / dist, [0.0]])

for grip, gname in [(CLOSE, "closed"), (OPEN, "open")]:
    env.reset(seed=3)
    cube_p = env.obj.pose.p[0].cpu().numpy()
    goal_p = env.goal_region.pose.p[0].cpu().numpy()
    d = goal_p[:2] - cube_p[:2]
    dist = float(np.linalg.norm(d))
    dirv = np.concatenate([d / dist, [0.0]])
    # 夹爪闭合时指尖并拢成"推杆"，指尖柱中心对准方块中心
    z_push = cube_p[2] + 0.0125
    back = cube_p - dirv * 0.08 + np.array([0, 0, z_push - cube_p[2]])
    back[2] = z_push
    move(back + np.array([0, 0, 0.15]), grip, label=f"hover({gname})")
    move(back, grip, tol=0.01, label=f"approach({gname})")
    target = cube_p + dirv * (dist + 0.05)
    target[2] = z_push
    move(target, grip, label=f"push({gname})")
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(7, dtype="float32"))
    cp = env.obj.pose.p[0].cpu().numpy()
    print(f"[{gname}] final cube={np.round(cp,3)} dist2d={np.linalg.norm(cp[:2]-goal_p[:2]):.3f} "
          f"success={info.get('success')}", flush=True)
