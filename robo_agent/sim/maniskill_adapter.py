"""ManiSkill 仿真执行器：Slaver 技能的真实执行后端。

与 mcp.ClientSession.call_tool 同接口（async call_tool(name, arguments)）。
技能 = 基于真值姿态反馈的脚本化策略（pd_ee_delta_pose 末端控制），
成功判定用仿真真值 info["success"] / is_grasped，不是硬编码。

技能 → 仿真任务映射：
    transport_cube(color)  → PickCube-v1：抓起方块放到目标区
    stack_cube()           → StackCube-v1：把方块叠到另一个上（Phase 2）
    push_cube()            → PushCube-v1：把方块推到目标区（Phase 2）

运行要求：WSL 无 GPU Vulkan，须 CPU 后端 + lavapipe：
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
（本模块 import 时自动设置）
"""
from __future__ import annotations

import os

# SAPIEN 在 import 时读取 ICD 配置，必须在导入 gym/mani_skill 之前设置
_LVP = "/usr/share/vulkan/icd.d/lvp_icd.json"
if os.path.exists(_LVP):
    os.environ.setdefault("VK_ICD_FILENAMES", _LVP)

import asyncio  # noqa: E402

import numpy as np  # noqa: E402

_TASKS = {
    "transport_cube": "PickCube-v1",
    "stack_cube": "StackCube-v1",
    "push_cube": "PushCube-v1",
}


class _ToolResult:
    """与 mcp.types.CallToolResult 结构兼容（content[0].text / isError）。"""

    def __init__(self, text: str, success: bool | None = None):
        self.content = [type("C", (), {"text": text})()]
        self.isError = (not success) if success is not None else False


class ManiSkillExecutor:
    """每个技能一个常驻仿真环境，重复调用时换 seed 重新布局。"""

    def __init__(self):
        self._envs: dict[str, object] = {}
        self._seed = 0

    def _get_env(self, skill: str):
        if skill not in _TASKS:
            raise ValueError(f"unknown skill: {skill}")
        if skill not in self._envs:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401

            self._envs[skill] = gym.make(
                _TASKS[skill], obs_mode="state", render_mode="rgb_array",
                control_mode="pd_ee_delta_pose",
                sim_backend="cpu", render_backend="cpu",
            )
        return self._envs[skill]

    async def call_tool(self, name: str, arguments: dict) -> _ToolResult:
        # 仿真步进是 CPU 密集计算，放到线程池避免阻塞事件循环
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute, name, arguments)

    # ------------------------------------------------------------------ #
    def _execute(self, name: str, arguments: dict) -> _ToolResult:
        if name == "transport_cube":
            ok, detail = self._pick_cube()
            return _ToolResult(f"PickCube-v1: {detail}, sim_success={ok}", success=ok)
        if name == "stack_cube":
            ok, detail = self._stack_cube()
            return _ToolResult(f"StackCube-v1: {detail}, sim_success={ok}", success=ok)
        if name == "push_cube":
            ok, detail = self._push_cube()
            return _ToolResult(f"PushCube-v1: {detail}, sim_success={ok}", success=ok)
        return _ToolResult(f"unknown skill: {name}", success=False)

    # ------------------------------------------------------------------ #
    # 基础控制原语（pd_ee_delta_pose）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tcp_pos(env) -> np.ndarray:
        return env.agent.tcp.pose.p[0].cpu().numpy()

    # Panda 夹爪关节范围 [0, 0.04]，0 = 闭合：action=+1 开，-1 合
    GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0

    def _move_towards(self, env, target: np.ndarray, gripper: float,
                      max_steps: int = 150, tol: float = 0.015) -> bool:
        """比例控制移动到目标位置（保持当前姿态），到达或步数耗尽即停。
        返回是否到达（考虑机械臂工作空间限制，未到达也停在最近处）。"""
        for _ in range(max_steps):
            pos = self._tcp_pos(env)
            delta = target - pos
            if np.linalg.norm(delta) < tol:
                return True
            action = np.zeros(7, dtype="float32")
            action[:3] = np.clip(delta * 5.0, -1.0, 1.0)
            action[6] = gripper
            env.step(action)
        return False

    def _grip(self, env, close: bool, steps: int = 25) -> None:
        action = np.zeros(7, dtype="float32")
        action[6] = self.GRIP_CLOSE if close else self.GRIP_OPEN
        for _ in range(steps):
            env.step(action)

    # ------------------------------------------------------------------ #
    # 技能：抓起方块放到目标区
    # ------------------------------------------------------------------ #
    def _pick_cube(self, seed: int | None = None) -> tuple[bool, str]:
        env = self._get_env("transport_cube").unwrapped
        self._seed += 1
        env.reset(seed=seed or self._seed)

        cube_p = env.cube.pose.p[0].cpu().numpy()
        goal_p = env.goal_site.pose.p[0].cpu().numpy()

        # 1) 移到方块上方（夹爪张开）
        self._move_towards(env, cube_p + np.array([0, 0, 0.15]), gripper=self.GRIP_OPEN)
        # 2) 下降到方块高度（panda tcp 可达最低 ~0.027，手指正好侧向包络方块）
        self._move_towards(env, cube_p + np.array([0, 0, 0.0]),
                           gripper=self.GRIP_OPEN, tol=0.008)
        # 3) 闭合夹爪
        self._grip(env, close=True)
        # 4) 抬起并验证抓取
        self._move_towards(env, cube_p + np.array([0, 0, 0.20]), gripper=self.GRIP_CLOSE)
        _, _, _, _, info = env.step(np.zeros(7, dtype="float32"))
        if not info.get("is_grasped", False):
            return False, "grasp failed (cube not held)"

        # 5) 运到目标上方，再把方块中心送进球内（手持触发判定）
        self._move_towards(env, goal_p + np.array([0, 0, 0.12]), gripper=self.GRIP_CLOSE)
        self._move_towards(env, goal_p + np.array([0, 0, 0.004]),
                           gripper=self.GRIP_CLOSE, tol=0.01)
        # 6) 保持闭合等待成功判定触发
        success = False
        for _ in range(6):
            self._grip_hold(env)
            _, _, _, _, info = env.step(np.zeros(7, dtype="float32"))
            if info.get("success"):
                success = True
                break
        # 7) 释放、抬开、稳定（注意：成功判定在第 6 步手持时已触发，
        #    释放后方块落回桌面，不能再读最终帧的 success）
        self._grip(env, close=False)
        self._move_towards(env, goal_p + np.array([0, 0, 0.20]), gripper=self.GRIP_OPEN)
        for _ in range(10):
            env.step(np.zeros(7, dtype="float32"))
        return bool(success), "pick and place finished"

    def _grip_hold(self, env, steps: int = 5) -> None:
        action = np.zeros(7, dtype="float32")
        action[6] = self.GRIP_CLOSE
        for _ in range(steps):
            env.step(action)

    # ------------------------------------------------------------------ #
    def _stack_cube(self) -> tuple[bool, str]:
        return False, "stack_cube not implemented yet"

    def _push_cube(self) -> tuple[bool, str]:
        env = self._get_env("push_cube").unwrapped
        self._seed += 1
        env.reset(seed=self._seed)

        cube_p = env.obj.pose.p[0].cpu().numpy()
        goal_p = env.goal_region.pose.p[0].cpu().numpy()
        d = goal_p[:2] - cube_p[:2]
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return True, "cube already at goal"
        dirv = np.concatenate([d / dist, [0.0]])

        # 夹爪闭合当"推杆"（张开的夹爪会从方块两侧穿过去推不到），
        # 指尖柱（长 0.025）中心对准方块中心
        z_push = cube_p[2] + 0.0125
        back = cube_p - dirv * 0.08
        back[2] = z_push
        self._move_towards(env, back + np.array([0, 0, 0.15]), gripper=self.GRIP_CLOSE)
        self._move_towards(env, back, gripper=self.GRIP_CLOSE, tol=0.01)
        target = cube_p + dirv * (dist + 0.05)
        target[2] = z_push
        self._move_towards(env, target, gripper=self.GRIP_CLOSE, max_steps=400, tol=0.02)
        for _ in range(10):
            _, _, _, _, info = env.step(np.zeros(7, dtype="float32"))
        return bool(info.get("success", False)), "push finished"

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()


if __name__ == "__main__":
    import sys

    skill = sys.argv[1] if len(sys.argv) > 1 else "transport_cube"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ex = ManiSkillExecutor()

    async def _main():
        ok = 0
        for i in range(n):
            r = await ex.call_tool(skill, {})
            good = not r.isError
            ok += good
            print(f"[{i+1}/{n}] {r.content[0].text}", flush=True)
        print(f"success rate: {ok}/{n}", flush=True)

    asyncio.run(_main())
    ex.close()
