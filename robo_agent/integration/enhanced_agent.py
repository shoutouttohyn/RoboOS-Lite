"""RoboOS 接入层：在不动 RoboOS 源码的前提下，以子类化方式注入两大改进模块。

注入点（对应 RoboOS 源码位置）：
  ① 规划前 — planner.forward() 构建 prompt 时（master/agents/planner.py:74）
     注入 RAG 检索结果 + 长期记忆召回，让 Planner "知道有哪些工具、
     记得以前怎么失败过"。
  ② 任务结束 — publish_global_task() 的全部子任务回收后（master/agents/agent.py）
     由 Master 总结 (任务, 子任务序列, 成败, 失败原因) 写入长期记忆。

用法：
    agent = EnhancedGlobalAgent("config/master.yaml",
                                rag=RAGEngine(), memory=MemoryManager())
    result = agent.run_task_sync("把红色方块放进碗里", config_flags={"rag": True, "memory": True})
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from typing import Dict

import sys
from pathlib import Path

# RoboOS master 目录加入 import path（内含 agents/ 包）
_ROBOOS_MASTER = Path(__file__).resolve().parent.parent.parent / "RoboOS" / "master"
if str(_ROBOOS_MASTER) not in sys.path:
    sys.path.insert(0, str(_ROBOOS_MASTER))

from agents.agent import GlobalAgent
from agents.planner import GlobalTaskPlanner
from agents.prompts import MASTER_PLANNING_PLANNING


def load_master_config(path: str) -> str:
    """把 yaml 里的 ENV:XXX 占位符替换为 .env 中的真实值，写回临时文件。
    返回可直接传给 GlobalAgent 的 config 路径。"""
    import os
    import re
    import tempfile

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    text = Path(path).read_text(encoding="utf-8")

    def _sub(m):
        return os.environ.get(m.group(1), m.group(0))

    text = re.sub(r'"ENV:([A-Z0-9_]+)"', _sub, text)
    tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_text(text, encoding="utf-8")
    return str(tmp)


class ContextInjectingPlanner(GlobalTaskPlanner):
    """注入 RAG + 记忆召回的规划器。复用父类全部逻辑，只扩展 prompt。"""

    def __init__(self, config, rag=None, memory=None):
        super().__init__(config)
        self.rag = rag
        self.memory = memory

    def forward(self, task: str, extra_context: str = "") -> str:
        """与父类签名兼容；extra_context 由 EnhancedGlobalAgent 组装后传入。"""
        if extra_context:
            task = f"{extra_context}\n\n# 当前任务: {task}"
        return super().forward(task)


class EnhancedGlobalAgent(GlobalAgent):
    """带记忆与 RAG 的 Master Agent。同步执行版本服务于评测框架。"""

    def __init__(self, config_path: str, rag=None, memory=None):
        # 手动构造，绕过父类 __init__ 中 planner 的直接实例化
        self._init_config(config_path)
        self._init_logger(self.config["logger"])
        from flag_scale.flagscale.agent.collaboration import Collaborator
        self.collaborator = Collaborator.from_config(self.config["collaborator"])
        self.planner = ContextInjectingPlanner(self.config, rag=rag, memory=memory)
        self.rag = rag
        self.memory = memory
        self.logger.info(f"Enhanced configuration loaded from {config_path}")
        self._init_scene(self.config["profile"])
        self._start_listener()
        self._results: Dict[str, dict] = {}
        self._results_lock = threading.Lock()
        self._registered: set = set()

    def wait_for_robots(self, timeout_s: int = 120) -> list:
        """规划前必须等到至少一个机器人注册（注册可能被外部 flush 掉）。"""
        import time as _time
        t0 = _time.time()
        while _time.time() - t0 < timeout_s:
            names = self.collaborator.read_all_agents_name() or []
            if names:
                return list(names)
            _time.sleep(2)
        return []

    def _ensure_result_listeners(self) -> None:
        """补注册：Slaver 若在 Master 订阅 AGENT_REGISTRATION 之前注册，
        注册事件会被漏掉，导致结果通道无人监听、Master 永远等不到回执。"""
        for name in self.collaborator.read_all_agents_name() or []:
            if name not in self._registered:
                self._handle_register(name)
                self._registered.add(name)

    # ---------- 钩子 ①：规划前组装上下文 ----------
    def _gather_context(self, task: str, use_rag: bool, use_memory: bool) -> str:
        parts = []
        if use_rag and self.rag:
            ctx = self.rag.context_for(task)
            if ctx:
                parts.append(ctx)
        if use_memory and self.memory:
            recalled = self.memory.recall(task)
            if recalled:
                parts.append(recalled)
        return "\n\n".join(parts)

    # ---------- 钩子 ②：结果回收时记录 ----------
    def _handle_result(self, data: str):
        super()._handle_result(data)
        try:
            d = json.loads(data)
            if d.get("subtask_handle"):
                with self._results_lock:
                    self._results[d["subtask_handle"]] = d
        except Exception:
            pass

    @staticmethod
    def _extract_json(input_string):
        """增强版 JSON 提取：RoboBrain 会输出 ```json 代码块，DeepSeek 常直接输出裸 JSON。
        依次尝试：代码块标记 → 整体解析 → 截取首个 { 到最后一个 } 再解析。"""
        start_marker, end_marker = "```json", "```"
        start_idx = input_string.find(start_marker)
        if start_idx != -1:
            end_idx = input_string.find(end_marker, start_idx + len(start_marker))
            if end_idx != -1:
                candidate = input_string[start_idx + len(start_marker):end_idx].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        try:
            return json.loads(input_string)
        except json.JSONDecodeError:
            pass
        brace_start, brace_end = input_string.find("{"), input_string.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(input_string[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                return None
        return None

    # ---------- 同步版任务执行（评测用） ----------
    def run_task_sync(self, task: str, config_flags: dict,
                      timeout_s: int = 600) -> dict:
        """规划 → 按序下发 → 回收结果 → 写记忆。返回 Episode 所需的全部字段。"""
        robots = self.wait_for_robots()
        if not robots:
            return {"task": task, "success": False, "failure_reason": "no_robot_registered",
                    "subtasks": [], "n_steps": 0}
        # 规划重试：LLM 输出偶发不合法（编造未注册机器人/格式漂移），最多 3 次
        reasoning, response = None, None
        context = self._gather_context(task, config_flags.get("rag", False),
                                       config_flags.get("memory", False))
        for attempt in range(3):
            response = self.planner.forward(task, extra_context=context)
            reasoning = self._extract_json(response)
            if self.reasoning_and_subtasks_is_right(reasoning):
                break
        if not self.reasoning_and_subtasks_is_right(reasoning):
            return {"task": task, "success": False, "failure_reason": "planning_illegal",
                    "subtasks": [], "n_steps": 0}

        subtasks = reasoning["subtask_list"]
        grouped = self._group_tasks_by_order(subtasks)
        task_id = str(uuid.uuid4()).replace("-", "")
        n_results_before = len(self._results)

        # 关键：下发前先确保已注册机器人的结果通道都被监听
        self._ensure_result_listeners()

        # 复用父类异步下发逻辑，但同步等待完成
        asyncio.run(self._dispath_subtasks_async(task, task_id, grouped, False))

        new_results = list(self._results.values())[n_results_before:]
        success = self._judge_success(new_results, grouped)
        failure_reason = "" if success else self._summarize_failure(new_results)

        # 钩子 ②：写入长期记忆
        if config_flags.get("memory", False) and self.memory:
            from ..memory.memory_manager import Episode
            self.memory.write(Episode(
                task=task,
                subtasks=[s["subtask"] for s in subtasks],
                success=success,
                failure_reason=failure_reason,
            ))

        return {"task": task, "success": success, "failure_reason": failure_reason,
                "subtasks": [s["subtask"] for s in subtasks],
                "n_steps": len(subtasks),
                "reasoning": reasoning.get("reasoning_explanation", "")}

    @staticmethod
    def _judge_success(results: list, grouped: dict) -> bool:
        """结果判定：全部子任务都收到非 ERROR 回传才成功。
        仿真适配层接入后，subtask_result 将携带 {"success": bool} 结构。"""
        total = sum(len(g) for g in grouped.values())
        ok = sum(1 for r in results
                 if r.get("subtask_result")
                 and not str(r["subtask_result"]).startswith("ERROR"))
        return ok >= total and total > 0

    @staticmethod
    def _summarize_failure(results: list) -> str:
        for r in results:
            res = r.get("subtask_result")
            if isinstance(res, dict) and not res.get("success"):
                return str(res.get("reason", "unknown"))
        return "partial_completion"
