"""抓取全流程探针 v2：镜像 adapter 的修正流程，逐步打印状态。
用法: PYTHONUNBUFFERED=1 python scripts/probe_pick_debug2.py
"""
import os

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array",
               control_mode="pd_ee_delta_pose",
               sim_backend="cpu", render_backend="cpu").unwrapped
env.reset(seed=7)
OPEN, CLOSE = 1.0, -1.0


def tcp():
    return env.agent.tcp.pose.p[0].cpu().numpy()


def step_info():
    _, _, _, _, i = env.step(np.zeros(7, dtype="float32"))
    return i


def move(target, gripper, max_steps=200, tol=0.015, label=""):
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


cube_p = env.cube.pose.p[0].cpu().numpy()
goal_p = env.goal_site.pose.p[0].cpu().numpy()
print("cube:", np.round(cube_p, 3), "goal:", np.round(goal_p, 3), flush=True)

move(cube_p + [0, 0, 0.15], OPEN, label="hover")
move(cube_p + [0, 0, 0.0], OPEN, tol=0.008, label="descend")

# 夹爪闭合过程逐段观察
for k in range(5):
    a = np.zeros(7, dtype="float32"); a[6] = CLOSE
    for _ in range(5):
        env.step(a)
    info = step_info()
    print(f"  closing {k}: tcp_z={tcp()[2]:.3f} is_grasped={info.get('is_grasped')} "
          f"grip_q={np.round(env.agent.robot.get_qpos().cpu().numpy()[-2:],3)}", flush=True)

move(cube_p + [0, 0, 0.20], CLOSE, label="lift")
info = step_info()
print("after lift: is_grasped=", info.get("is_grasped"), "cube_z=", f"{env.cube.pose.p[0].cpu().numpy()[2]:.3f}", flush=True)

move(goal_p + [0, 0, 0.12], CLOSE, label="to_goal")
# 目标是悬浮球：把方块中心送进球内（手持），成功判定自动触发
move(goal_p + [0, 0, 0.004], CLOSE, tol=0.01, label="to_goalsphere")
ok = False
for k in range(6):
    a = np.zeros(7, dtype="float32"); a[6] = CLOSE
    for _ in range(5):
        env.step(a)
    info = step_info()
    cp = env.cube.pose.p[0].cpu().numpy()
    print(f"  hold {k}: cube={np.round(cp,3)} success={info.get('success')}", flush=True)
    if info.get("success"):
        ok = True
        break
if ok:
    a = np.zeros(7, dtype="float32"); a[6] = OPEN
    for _ in range(30):
        env.step(a)
    move(goal_p + [0, 0, 0.20], OPEN, label="retreat")
    for _ in range(10):
        info = step_info()
cp = env.cube.pose.p[0].cpu().numpy()
print(f"final: ok={ok} cube={np.round(cp,3)} goal={np.round(goal_p,3)}", flush=True)
