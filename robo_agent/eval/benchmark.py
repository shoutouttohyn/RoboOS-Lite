"""消融评测框架：Baseline / +RAG / +Memory 三组配置，统一指标输出。

用法（仿真适配层 ready 之后）：
    python -m eval.benchmark --config baseline --episodes 10
    python -m eval.benchmark --config rag --episodes 10
    python -m eval.benchmark --config full --episodes 10
    python -m eval.report                        # 汇总三组结果出对比表
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# 5 个任务：由 ManiSkill 仿真技能组合而成（成功判定 = 仿真真值 success）
TASK_SUITE = [
    {"id": "t1", "name": "transport_red", "instruction": "把红色方块抓取并放到目标区域",
     "skills": ["transport_cube"]},
    {"id": "t2", "name": "transport_green", "instruction": "把绿色方块运输到目标区",
     "skills": ["transport_cube"]},
    {"id": "t3", "name": "transport_blue", "instruction": "将蓝色方块搬运到目标位置",
     "skills": ["transport_cube"]},
    {"id": "t4", "name": "transport_far", "instruction": "把红色方块抓取并放到较远的的目标区域",
     "skills": ["transport_cube"]},
    {"id": "t5", "name": "push_to_goal", "instruction": "把方块推到目标区域",
     "skills": ["push_cube"]},
]

# 三组消融配置：开关即 JD 里"评估体系的建立与持续调优"的证据
CONFIGS = {
    "baseline": {"rag": False, "memory": False},
    "rag":      {"rag": True,  "memory": False},
    "full":     {"rag": True,  "memory": True},
}


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    n_steps: int
    latency_s: float
    tokens: int = 0
    plan_legal: bool | None = None   # 人工标注位：子任务序列是否合法
    failure_reason: str = ""
    memory_hits: int = 0             # 本局触发的记忆召回条数


@dataclass
class RunStats:
    config: str
    n_episodes: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_latency_s: float = 0.0
    total_tokens: int = 0
    memory_hit_rate: float = 0.0     # 有可用历史时召回被采纳的比例
    episodes: list[EpisodeResult] = field(default_factory=list)


class BenchmarkRunner:
    """驱动整个评测。executor 是仿真适配层提供的 callable：
    executor(instruction, config) -> EpisodeResult
    """

    def __init__(self, executor, config_name: str):
        self.executor = executor
        self.config_name = config_name
        self.stats = RunStats(config=config_name)

    def run(self, episodes_per_task: int = 10) -> RunStats:
        for task in TASK_SUITE:
            for i in range(episodes_per_task):
                t0 = time.time()
                r = self.executor(task["instruction"], CONFIGS[self.config_name])
                r.task_id = task["id"]
                r.latency_s = time.time() - t0
                self.stats.episodes.append(r)
        self._aggregate()
        self._save()
        return self.stats

    def _aggregate(self) -> None:
        eps = self.stats.episodes
        n = len(eps)
        self.stats.n_episodes = n
        self.stats.success_rate = sum(e.success for e in eps) / n
        self.stats.avg_steps = sum(e.n_steps for e in eps) / n
        self.stats.avg_latency_s = sum(e.latency_s for e in eps) / n
        self.stats.total_tokens = sum(e.tokens for e in eps)
        usable = [e for e in eps if e.memory_hits >= 0]
        self.stats.memory_hit_rate = sum(e.memory_hits for e in usable) / n

    def _save(self) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{self.config_name}.json"
        path.write_text(json.dumps(asdict(self.stats), indent=2, ensure_ascii=False))
        print(f"saved -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    print(f"配置 {args.config}: {CONFIGS[args.config]}")
    print("注意：需要先实现 executor（ManiSkill 仿真适配层），见 eval/README")
