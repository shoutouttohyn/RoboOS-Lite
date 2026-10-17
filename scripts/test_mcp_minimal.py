"""MCP stdio 最小复现：直连 demo_robot_local/skill.py，逐步验证。
用法: python scripts/test_mcp_minimal.py
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SKILL = Path(__file__).resolve().parent.parent / "run" / "slaver" / "demo_robot_local" / "skill.py"


async def main():
    print("[1] starting stdio client...", flush=True)
    params = StdioServerParameters(command=sys.executable, args=[str(SKILL)])
    async with stdio_client(params) as (read, write):
        print("[2] transport up, creating session...", flush=True)
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[3] session initialized, listing tools...", flush=True)
            tools = await session.list_tools()
            print(f"[4] tools: {[t.name for t in tools.tools]}", flush=True)
            print("[5] calling navigate_to_target...", flush=True)
            result = await asyncio.wait_for(
                session.call_tool("navigate_to_target", {"target": "kitchen table"}),
                timeout=20)
            print(f"[6] tool result: {result.content[0].text}", flush=True)
            print("[7] ALL OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
