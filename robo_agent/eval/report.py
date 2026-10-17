"""汇总多组消融结果，输出 Markdown 对比表（直接贴 README/简历）。"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def report() -> str:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        s = json.loads(f.read_text())
        rows.append((
            s["config"], s["n_episodes"], f"{s['success_rate']:.1%}",
            f"{s.get('avg_steps', 0):.1f}", f"{s.get('avg_latency_s', 0):.1f}s",
            s.get("total_tokens", "-"), f"{s.get('memory_hit_rate', 0):.1%}",
        ))
    header = "| 配置 | 局数 | 任务成功率 | 平均步数 | 平均延迟 | Token 总量 | 记忆命中率 |"
    sep = "|---|---|---|---|---|---|---|"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    table = "\n".join([header, sep, body])
    print(table)
    return table


if __name__ == "__main__":
    report()
