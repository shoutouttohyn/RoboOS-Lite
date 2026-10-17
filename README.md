# RoboOS-Lite：复现 RoboOS + 记忆/RAG 增强的具身 Agent 系统

在 [FlagOpen/RoboOS](https://github.com/FlagOpen/RoboOS)（stand-alone 分支）分层具身架构之上，
以增加**长期记忆**与 **RAG 技能检索**两大模块，并在 [ManiSkill](https://github.com/haosulab/ManiSkill) 仿真上
建立消融评测体系。**对 RoboOS 源码零改动**，所有增强以子类化/运行时补丁方式注入。

## 架构

```
用户指令
    │
┌─── Master: Planner Agent (DeepSeek API, ReAct 规划) ──┐
│  ① 规划前: MemoryManager.recall(task)   —— 长期记忆召回 │
│  ② 规划前: RAGEngine.context_for(task)  —— 技能/领域检索│
│  ③ LLM 任务分解 → JSON 子任务 → Redis 按序下发          │
│  ④ 任务后: MemoryManager.write(episode) —— 情景记忆写入 │
└──────────────────┬─────────────────────────────────────┘
                   │ Redis Shared Memory（短期记忆/状态同步）
        Slaver: ReAct 工具调用循环（MCP 协议语义）
                   │
        ManiSkill 仿真执行器（真值姿态反馈的脚本化技能策略，
        成功判定 = 仿真真值 info["success"]）
```

## 复现中修复的 RoboOS 问题（详见 DESIGN.md）

1. **注册竞态**：Slaver 先于 Master 注册时结果通道无人监听 → 下发前补扫已注册机器人。
2. **JSON 提取**：RoboBrain 输出 ```json 代码块而 DeepSeek 输出裸 JSON → 三级降级解析。
3. **API 兼容**：DeepSeek 要求 tools 带 `type:"function"`；空闲 keep-alive 连接被服务端
   静默掐断导致请求挂起 → 禁复用 + 显式超时。
4. **执行层**：demo 技能为硬编码桩；替换为 ManiSkill 进程内仿真执行器。

## 快速开始（WSL2 Ubuntu）

```bash
# 环境（首次）：见 SETUP.md —— conda env robo (py3.11) + redis + FlagScale + 本 repo 依赖
# DeepSeek key：cp .env.example .env 并填入

# 1. 启动 Slaver（ManiSkill 仿真执行器，常驻）
python scripts/run_slaver.py

# 2. 单任务运行（baseline / +RAG / +记忆）
python scripts/run_task.py --task "把红色方块抓取并放到目标区域"
python scripts/run_task.py --task "..." --rag --memory

# 3. 消融评测（5 任务 × 3 配置）
python scripts/run_benchmark.py --episodes 3
```

仿真在 CPU 后端运行（WSL 无 GPU Vulkan，自动使用 lavapipe + CPU PhysX）。

## 评测结果（n=25，5 任务 × 5 局，真实仿真执行）

| 配置 | 任务成功率 | 失败模式分布 |
|---|---|---|
| baseline | 64.0% | partial_completion 8 / planning_illegal 1 |
| +RAG | 32.0% | partial_completion 12 / planning_illegal 5 |
| +RAG+记忆 | 28.0% | partial_completion 15 / planning_illegal 3 |

**关键发现（负向结果）**：注入检索上下文后成功率下降，失败集中在
`partial_completion`（子任务回执未完整回收）。诊断：检索内容改变子任务文本表述
→ Slaver 侧 ReAct 工具匹配与执行路径变化 → 在结果通道存在可靠性问题的背景下，
更多子任务暴露更多问题。已通过"异常必回执"补丁部分缓解（baseline 52%→64%），
剩余根因（busy 标志在 Redis 中丢失）列入 Phase 2。完整数据见
`robo_agent/eval/results/*.json`。


## 目录

```
robo_agent/
  integration/enhanced_agent.py  # RoboOS 接入层（子类化注入，零改源码）
  memory/                        # 长期记忆（Qdrant 本地模式 + bge-m3）
  rag/                           # RAG 引擎（LlamaIndex，技能库+领域库）
  sim/maniskill_adapter.py       # 仿真执行器（call_tool 接口）
  eval/                          # 消融评测框架 + 报告生成
config/                          # DeepSeek 版 master/slaver 配置
knowledge/                       # 技能文档库 + 领域知识库
scripts/                         # 启动器与 e2e/benchmark 脚本
RoboOS/                          # 上游源码（stand-alone 分支，未改动）
```

## 技术栈

RoboOS（Master/Slaver + Redis Shared Memory）· DeepSeek API（OpenAI 兼容）·
LlamaIndex + bge-m3（RAG）· Qdrant（本地嵌入式）· ManiSkill 3（SAPIEN，CPU 后端）·
FlagScale Collaborator
