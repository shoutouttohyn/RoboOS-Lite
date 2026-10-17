"""最小复现：复刻 Slaver 的并发结构 —— listener 线程 run_coroutine_threadsafe + 主 asyncio 循环 + MCP stdio。
用法: python scripts/test_repro_hang.py
若挂起: kill -USR1 <pid> 后查看栈
"""
import asyncio
import faulthandler
import json
import os
import signal
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SKILL = str(Path(__file__).resolve().parent.parent / "run" / "slaver" / "demo_robot_local" / "skill.py")

faulthandler.register(signal.SIGUSR1, all_threads=True)


class FakeModel:
    """模拟 OpenAIServerModel：同步阻塞返回 tool_call"""
    calls = 0

    def __call__(self, **kwargs):
        FakeModel.calls += 1
        print(f"[model] sync call #{FakeModel.calls}", flush=True)
        return {
            "role": "assistant",
            "content": "I'll navigate to the kitchen table.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "navigate_to_target", "arguments": '{"target": "kitchen table"}'},
            }],
        }


class MiniAgent:
    def __init__(self, session, model):
        self.session = session
        self.model = model

    async def run(self, task):
        for step in range(3):
            print(f"[agent] step {step}: calling model", flush=True)
            msg = self.model(task=task)          # 同步阻塞调用（复刻真实结构）
            print(f"[agent] step {step}: got tool_call {msg['tool_calls'][0]['function']['name']}", flush=True)
            result = await self.session.call_tool(
                msg["tool_calls"][0]["function"]["name"],
                json.loads(msg["tool_calls"][0]["function"]["arguments"]))
            print(f"[agent] step {step}: tool result: {result.content[0].text}", flush=True)
        return "done"


def main():
    loop_holder = {}

    def listener():
        """复刻 collaborator.listen 线程：收到消息后 run_coroutine_threadsafe + future.result()"""
        asyncio.run_coroutine_threadsafe(handle_task({"task": "Navigate to the kitchen table"}),
                                         loop_holder["loop"]).result()
        print("[listener] task completed", flush=True)

    async def handle_task(data):
        print("[handle_task] start", flush=True)
        agent = MiniAgent(session_holder["session"], FakeModel())
        result = await agent.run(data["task"])
        print(f"[handle_task] result={result}", flush=True)

    session_holder = {}

    async def amain():
        loop_holder["loop"] = asyncio.get_running_loop()
        # 复刻 connect_to_robot：主循环上建立 MCP stdio session
        params = StdioServerParameters(command=sys.executable, args=[SKILL])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"[main] MCP ready, tools={[t.name for t in tools.tools]}", flush=True)
                session_holder["session"] = session
                t = threading.Thread(target=listener, daemon=True)
                t.start()
                while True:
                    await asyncio.sleep(1)

    asyncio.run(amain())


if __name__ == "__main__":
    main()
