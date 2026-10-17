# RoboOS-Lite：复现 RoboOS + 记忆/RAG 增强的具身 Agent 项目

> 目标：复现 FlagOpen/RoboOS 的分层具身架构，在其上增加**长期记忆**与 **RAG 技能检索**两大模块，
> 在 ManiSkill 仿真上建立消融评测体系，匹配「Agent算法实习生-灵犀」JD。

## 设计原则

1. **最小侵入**：保留 RoboOS 原生 Master/Slaver + Redis 架构，改进以"外挂模块"形式接入 Master 的规划流程。
2. **仿真优先**：无真机，Slaver 技能底层用 ManiSkill 仿真适配器替代真机 SDK。
3. **先跑通后调优**：Phase 1 用 API 大模型（DeepSeek/Qwen）+ 纯向量检索；本地 VLM、rerank、混合检索留到 Phase 2。

## 系统架构

```
用户指令（文本）
    │
┌─── Master: Planner Agent (LLM, ReAct) ─────────────┐
│  ① 规划前：MemoryManager.recall(task)               │   改进模块（新增）
│     ② 规划前：RAG.query("可用技能/领域知识")          │   改进模块（新增）
│  ③ LLM 任务分解 → 子任务下发                         │   RoboOS 原生
│  ④ 执行后：MemoryManager.write(任务摘要+成败原因)     │   改进模块（新增）
└──────────────────┬─────────────────────────────────┘
                   │ 子任务
        Redis Shared Memory（原生，短期记忆/状态同步）
                   │
        Slaver × N：技能执行（原生流程 + 仿真适配层）
                   │
        ManiSkill 仿真环境（PickCube / StackCube / PushCube）
```

## 改进模块设计

### ① 长期记忆（MemoryManager）
- 短期：复用 Redis Shared Memory（对话上下文 + 执行状态）。
- 长期：Qdrant 本地容器 + bge-m3 embedding（CPU 可跑）。
- 写入触发：任务结束后由 Master 总结 `(任务, 子任务序列, 成败, 失败原因)` 写入情景记忆集合。
- 召回触发：新任务规划前，按任务文本相似度检索 Top-3 历史经验注入 prompt。

### ② RAG 引擎（LlamaIndex）
- 库 A「技能文档库」：每个技能一篇 markdown（名称/参数/前置条件/失败模式），让 LLM 知道何时调用哪个工具。
- 库 B「领域知识库」：工业/家庭操作手册语料（Phase 1 先用 20~30 篇公开文档）。
- Phase 1：纯向量检索 Top-K 注入规划 prompt；Phase 2 加 BM25 混合检索 + bge-reranker。

### ③ 评估体系
- 任务集：5 个长时程任务（如"整理桌面"= 检测→抓取→放置 × N，由基础技能组合）。
- 指标：任务成功率、子任务规划合法性（人工标注）、平均步数、端到端延迟、token 成本、记忆检索命中率。
- 消融：Baseline（原生 RoboOS）/ +RAG / +RAG+记忆，各跑 N=10 次。

## 技术栈

| 组件 | 选型 |
|---|---|
| Agent 骨架 | RoboOS（stone-alone 分支） |
| LLM | DeepSeek API / Qwen API（Phase 1）；本地 Qwen2.5-VL（Phase 2 可选） |
| 仿真 | ManiSkill（SAPIEN） |
| 向量库 | Qdrant（Docker） |
| Embedding | bge-m3（CPU） |
| RAG | LlamaIndex |
| 消息/短期记忆 | Redis |

## 源码侦察结论（2026-10-15，stand-alone @1c39b85）

- **技能层已是 MCP**：`slaver/demo_robot_local/skill.py` 是 FastMCP Server（navigate/grasp/place），
  Slaver 侧 `MultiStepAgent` 是 ReAct 循环 + MCP ClientSession。JD 的"MCP 工具调用"原生具备，
  我们只需把假技能换成 ManiSkill 仿真技能。
- **Master 规划入口**：`GlobalAgent.publish_global_task()` → `GlobalTaskPlanner.forward()`
  （prompt 模板 `MASTER_PLANNING_PLANNING`，4 个槽位：机器人列表/工具信息/场景/任务）
  → JSON 子任务 → Redis 按 `subtask_order` 分组下发 → `wait_agents_free` 等待。
- **接入方式（最小侵入）**：子类化注入，不改 RoboOS 源码。
  - `ContextInjectingPlanner(GlobalTaskPlanner)`：规划前把 RAG + 记忆召回拼进 task 槽位。
  - `EnhancedGlobalAgent(GlobalAgent)`：覆写 `_handle_result` 收集子任务结果，
    任务完成后写情景记忆。代码在 `robo_agent/integration/enhanced_agent.py`。
- **隐藏依赖**：`flag_scale.flagscale.agent.collaboration.Collaborator` 来自 FlagOpen/FlagScale
  （固定 commit 3fc2037），不在 requirements.txt 里，安装步骤见 SETUP.md §3。
- **demo 技能是桩**：`demo_robot_local/skill.py` 返回硬编码成功串，仅供演示；
  复现现象 ≠ 真实执行，评测必须走我们自己的仿真适配层。
- **注册竞态 bug（已修复）**：Master 只在收到 `AGENT_REGISTRATION` 频道消息时才监听
  各机器人的结果通道；Slaver 若先注册，事件被漏掉 → Master 永远等不到执行回执。
  修复：`EnhancedGlobalAgent._ensure_result_listeners()` 下发前补扫已注册机器人。
- **LLM 输出格式差异**：RoboBrain 输出 json 代码块，DeepSeek 常输出裸 JSON →
  原版 `_extract_json` 提取失败。修复：`EnhancedGlobalAgent._extract_json` 三级降级解析。
- **DeepSeek API 兼容性**：tools 定义必须带 `type: "function"`（RoboBrain 端点不校验）；
  且服务端会静默掐断空闲 keep-alive 连接，复用死连接导致请求永久挂起（CLOSE-WAIT），
  需禁用 keep-alive 复用 + 显式超时（见 scripts/run_slaver.py 补丁区）。
- **stdio 传输替换**：demo 阶段的 MCP stdio 子进程传输在单机环境有稳定性问题，
  已换为进程内执行器（接口与 `session.call_tool` 完全一致）；ManiSkill 适配层也将
  以进程内技能方式接入，MCP 协议语义（工具发现/调用/结果）不变。
- **模型接入**：Master 走 OpenAI 兼容接口（config `cloud_type: "default"`），
  默认指向本地 vLLM RoboBrain2.0-7B；换 DeepSeek 只需改 config，见 `config/master_deepseek.yaml`。

## Phase 划分

- **Phase 1（本周）**：环境（WSL2+Ubuntu）→ RoboOS 复现 → ManiSkill 适配 2~3 技能 → RAG + 记忆模块 → 5 任务三消融 → 写简历。
- **Phase 2（面试前）**：按 `notes/` 模块笔记补深度；可选补 Verifier 反思模块、rerank、本地 VLM。

## 风险

- RoboOS 官方云端 RoboBrain 需要 API 权限 → 换 DeepSeek/Qwen API，prompt 结构不变（已验证接口兼容）。
- Windows 原生跑不动 SAPIEN/Qdrant → 必须 WSL2（已就绪）；GPU Vulkan 在 WSL 不可用 →
  ManiSkill 用 `sim_backend="cpu"` + lavapipe（`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`，
  adapter 模块自动设置）。
- GitHub 直连被重置 → 统一走 ghproxy.net 镜像（SETUP.md 已注明）。
- RoboOS 2.0 文档未发布 → 以 stand-alone 代码为准。
- Qdrant 本地模式：存储目录必须在 WSL 原生文件系统（DrvFS /mnt/d 文件锁不可靠）；
  同进程多客户端按路径单例复用（`qdrant_client_factory._CLIENTS`）。
- llama-index 的 HuggingFaceEmbedding 默认用自有 cache 目录，找不到 HF 已下载模型 →
  显式传 `cache_folder=<HF_HOME>/hub`。
