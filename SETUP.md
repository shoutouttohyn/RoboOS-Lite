# 环境安装清单（Windows → WSL2）

> 预计耗时：3~6 小时（主要取决于网络和显卡）。全部命令在 WSL2 Ubuntu 终端内执行。
> 2026-10-15 实测更新：WSL2/GPU 已就绪，GitHub 直连不通，须走 ghproxy.net 镜像。

## 0. 一次性检查（已完成 ✅）

- WSL2 Ubuntu、GPU 透传（RTX 3070 8G）、Miniconda3 均已就绪。
- 系统 Python 3.14 太新，**必须**用 conda 环境（3.11）。

## 1. 基础依赖（在 WSL Ubuntu 终端执行）

```bash
sudo apt update && sudo apt install -y git redis-server build-essential
sudo service redis-server start
redis-cli ping   # PONG = 成功
conda create -n robo python=3.11 -y && conda activate robo
pip install -U pip
```

## 2. RoboOS 源码（Windows 侧已 clone 到 robo-agent/RoboOS，WSL 可直接访问）

GitHub 直连会 `Connection reset`，**必须**走镜像：

```bash
git clone --depth 1 --branch stand-alone https://ghproxy.net/https://github.com/FlagOpen/RoboOS.git
# 注意：分支名是 stand-alone（README 里的 stone-alone 是官方笔误）
```

## 3. FlagScale 依赖（RoboOS 的 Collaborator 来自这里，requirements.txt 里没有！）

```bash
git clone https://ghproxy.net/https://github.com/FlagOpen/FlagScale.git
cd FlagScale && git checkout 3fc2037f90917227bd4aebabd9d7b330523f437c
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation
python -c "from flag_scale.flagscale.agent.collaboration import Collaborator; print('ok')"
```

## 4. 本项目依赖

```bash
cd <robo-agent 目录>
pip install -r requirements.txt
cp .env.example .env   # 填入 DeepSeek API key
```

## 5. 各组件验证（按顺序）

| 组件 | 验证 | 通过标准 |
|---|---|---|
| Qdrant | `docker run -d -p 6333:6333 qdrant/qdrant`（无 Docker 则 `pip install qdrant-client` 本地模式） | `curl localhost:6333` 返回 JSON |
| Embedding | `python -m robo_agent.memory.memory_manager` | 打印召回结果 |
| RoboOS demo | 见 RoboOS README：master/run.py + slaver/run.py + gradio_ui | Web 界面能下发任务 |
| LLM API | `.env` 填 key 后调 deepseek-chat | 返回正常 |
| ManiSkill | `pip install mani-skill`，跑官方 PickCube demo | 能出视频/画面 |

## 6. 常见坑

- **SAPIEN/ManiSkill 装不上** → 必须 WSL2，别在 Windows 原生 Python 里试。
- **huggingface.co 连不上**（国内网络）→ 下载模型用 `HF_ENDPOINT=https://hf-mirror.com`；
  已缓存的模型用 `HF_HUB_OFFLINE=1` 完全离线加载。
- **FlagEmbedding 1.4.x + transformers 5.x 有 bug**（tokenizer.pad 收到 normalize_embeddings）→
  记忆模块已改用 sentence-transformers 加载 bge-m3，不依赖 FlagEmbedding。
- **conda 26.x 要先接受 ToS** → `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`（pkgs/r 同理）。
- **FlagScale 的 run 模块会泄漏到 sys.path** → 启动脚本用 importlib 按文件路径加载 RoboOS 的 slaver/run.py，不能直接 `import run`。
- **RoboOS requirements 会降级 openai/transformers 等** → 先装 RoboOS 的再装本项目的，或装完跑一遍导入冒烟。
- **RoboOS demo 技能是假的** → `slaver/demo_robot_local/skill.py` 返回硬编码成功串，这是 stand-alone 演示的设计；真实执行靠我们的 ManiSkill 适配层替换。
- **e2e 挂起排查三件套**（2026-10-15 实战）：① Master 漏听注册事件→结果通道无人订阅（已修，见 DESIGN.md）；② DeepSeek 需要 tools 带 `type:"function"`；③ DeepSeek 会掐死空闲 keep-alive 连接，必须禁复用+设超时。调试挂起进程：`kill -USR1 <slaver_pid>`（faulthandler 已注册）。
- **杀进程要杀干净**：TaskStop 只杀 Windows 侧包装，WSL 里的 python 会变僵尸（`pgrep -af` 检查）；本地 Qdrant 有文件锁，僵尸 Master 会阻塞新实例启动。
- **Qdrant 本地模式两个坑**：存储目录必须在 WSL 原生文件系统（`/mnt/d` DrvFS 锁不可靠，会幽灵锁；默认已改为 `~/.local/share/robo-agent/qdrant`）；同进程开第二个 client 会报 already accessed（工厂已按路径单例）。
- **ManiSkill 在 WSL**：GPU Vulkan 不可用 → `sim_backend="cpu"` + lavapipe（adapter 自动设置 `VK_ICD_FILENAMES`）；官方 mplib 运动规划在本环境 segfault，已改用自研脚本化策略（pd_ee_delta_pose + 真值姿态反馈）。
- **Panda 夹爪方向**：action=+1 开 / -1 合（关节 0.04→0）；PickCube 的目标是悬浮球，必须手持送进球内等判定触发，不能放到桌面。

## 7. 完成后

按 `DESIGN.md` Phase 1 推进，卡住把报错贴给我。
