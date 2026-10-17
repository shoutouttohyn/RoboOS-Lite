"""Slaver 启动器：不改动 RoboOS 源码/配置，在独立工作区运行 demo 机器人。

用法：python scripts/run_slaver.py
工作区：robo-agent/run/slaver/（config.yaml + demo_robot_local/ 拷贝 + images/）
"""
import asyncio
import os
import shutil
import sys
import threading
from pathlib import Path

ROBO_AGENT = Path(__file__).resolve().parent.parent
ROBOOS_SLAVER = ROBO_AGENT / "RoboOS" / "slaver"
WORKSPACE = ROBO_AGENT / "run" / "slaver"

sys.path.insert(0, str(ROBO_AGENT))
sys.path.insert(0, str(ROBOOS_SLAVER))  # 提供 agents/ tools/ run.py 模块


def prepare_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "images").mkdir(exist_ok=True)
    # demo 机器人（FastMCP 技能服务）拷入工作区，使相对路径可用
    src = ROBOOS_SLAVER / "demo_robot_local"
    dst = WORKSPACE / "demo_robot_local"
    if not dst.exists():
        shutil.copytree(src, dst)
    # 生成替换 ENV 占位符后的 config.yaml
    from robo_agent.integration.enhanced_agent import load_master_config
    cfg = load_master_config(str(ROBO_AGENT / "config" / "slaver_deepseek.yaml"))
    shutil.copy(cfg, WORKSPACE / "config.yaml")


def main() -> None:
    prepare_workspace()
    os.chdir(WORKSPACE)
    # 让 Slaver 派生的 MCP 子进程（command="python"）也用 robo 环境的 python
    env_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{env_bin}:{os.environ.get('PATH', '')}"

    # 调试挂起：kill -USR1 <pid> 把所有线程栈 dump 到 stderr
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    # 按文件路径显式加载，避免与 flag_scale 包泄漏到 sys.path 的顶层 run 模块撞名
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "roboos_slaver_run", ROBOOS_SLAVER / "run.py")
    slaver_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slaver_run)

    # DeepSeek API 严格要求 tools 定义带 type 字段（RoboBrain 的端点不需要），
    # 在 connect_to_robot 完成后补齐，保持 RoboOS 源码不动。
    _orig_connect = slaver_run.RobotManager.connect_to_robot

    async def _patched_connect(self):
        await _orig_connect(self)
        for tool in self.tools or []:
            tool.setdefault("type", "function")

    slaver_run.RobotManager.connect_to_robot = _patched_connect

    # DeepSeek 服务端会静默掐断空闲 keep-alive 连接，复用死连接会让请求永久挂起
    # （表现为 CLOSE-WAIT + epoll 空等）。禁用 keep-alive 复用 + 显式超时。
    from agents import models as slaver_models
    _orig_model_init = slaver_models.OpenAIServerModel.__init__

    def _patched_model_init(self, *args, **kwargs):
        import httpx
        ck = kwargs.get("client_kwargs") or {}
        ck.setdefault("timeout", 120)
        ck.setdefault("max_retries", 5)
        ck.setdefault("http_client", httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=0),
        ))
        kwargs["client_kwargs"] = ck
        _orig_model_init(self, *args, **kwargs)

    slaver_models.OpenAIServerModel.__init__ = _patched_model_init

    async def _local_connect(self):
        """复刻 connect_to_robot 的注册流程；技能由 ManiSkill 仿真执行器提供。"""
        from datetime import datetime
        import json as _json

        cfg = slaver_run.config
        self.tools = [
            {"function": {"name": "transport_cube",
                          "description": ("Pick up the colored cube and transport it to the goal area. "
                                          "Args: color (str, optional), e.g. 'red'")},
             "input_schema": {"type": "object", "properties": {"color": {"type": "string"}}},
             "type": "function"},
            {"function": {"name": "push_cube",
                          "description": ("Push the cube to the goal region without grasping. "
                                          "Args: color (str, optional)")},
             "input_schema": {"type": "object", "properties": {"color": {"type": "string"}}},
             "type": "function"},
            {"function": {"name": "stack_cube",
                          "description": "Stack one cube onto another. Args: color (str, optional)"},
             "input_schema": {"type": "object", "properties": {"color": {"type": "string"}}},
             "type": "function"},
        ]
        self.tool_matcher.fit(self.tools)
        self.session = ManiSkillExecutor()

        robot_name = cfg["robot"]["name"]
        self.robot_name = robot_name
        register = {"robot_name": robot_name, "robot_tool": self.tools,
                    "robot_state": "idle", "timestamp": int(datetime.now().timestamp())}
        with self.lock:
            self.collaborator.register_agent(robot_name, _json.dumps(register), expire_second=60)
            threading.Thread(target=self._heartbeat_loop, daemon=True,
                             args=(robot_name,), name=f"heartbeat_{robot_name}").start()
            channel_b2r = f"roboos_to_{robot_name}"
            threading.Thread(target=lambda: self.collaborator.listen(
                channel_b2r, self.handle_task), daemon=True, name=channel_b2r).start()
        print("[slaver] local in-process tools connected", flush=True)

    # 用 ManiSkill 仿真执行器替换 MCP stdio 传输：技能在仿真里真实执行，
    # 成功判定来自仿真真值。接口与 session.call_tool 完全一致。
    from robo_agent.sim.maniskill_adapter import ManiSkillExecutor

    slaver_run.RobotManager.connect_to_robot = _local_connect

    # 自愈：注册信息被外部 flush 后心跳会失败，此时自动重新注册
    _orig_hb = slaver_run.RobotManager._heartbeat_loop

    def _self_healing_hb(self, robot_name):
        import time as _time
        while not self._shutdown_event.is_set():
            try:
                ok = self.collaborator.agent_heartbeat(robot_name, seconds=60)
                if not ok:
                    import json as _json
                    from datetime import datetime
                    register = {"robot_name": robot_name, "robot_tool": self.tools,
                                "robot_state": "idle",
                                "timestamp": int(datetime.now().timestamp())}
                    self.collaborator.register_agent(robot_name, _json.dumps(register),
                                                     expire_second=60)
                    print(f"[slaver] re-registered {robot_name}", flush=True)
                _time.sleep(30)
            except Exception as e:
                if not self._shutdown_event.is_set():
                    print(f"Heartbeat error: {e}")
                break

    slaver_run.RobotManager._heartbeat_loop = _self_healing_hb

    # 可靠性：Slaver 执行抛异常时也必须回传结果（否则 Master 侧表现为
    # partial_completion，无法区分"技能失败"和"消息丢失"）
    _orig_execute_task = slaver_run.RobotManager._execute_task

    async def _robust_execute_task(self, task_data):
        try:
            return await _orig_execute_task(self, task_data)
        except Exception as e:
            print(f"[slaver] task error, sending error result: {e}", flush=True)
            self._send_result(robot_name=self.robot_name, task=task_data.get("task"),
                              task_id=task_data.get("task_id"),
                              result=f"ERROR: {e}", tool_call=[])

    slaver_run.RobotManager._execute_task = _robust_execute_task

    asyncio.run(slaver_run.main())


if __name__ == "__main__":
    main()
