# 01. 软件需求规格说明书 (SRS)

- **文档版本**：v1.1
- **适用阶段**：MVP（兼顾本地微调闭环与多任务/大数据集/无标签扩展）
- **目标实测环境**：
  - OS / Python：Linux x86_64 / Python 3.11+ / Docker
  - GPU：6 × NVIDIA RTX 5090（32 GB VRAM/卡，Blackwell `sm_120`，CUDA 13.2）；**当前实时空闲 1 卡（GPU 1，32 GB）**
  - CPU / RAM：AMD EPYC 9554（120+ 在线核）/ 1.0 TB RAM（~906 GB 可用）
  - 存储约束：系统盘 `/` 仅剩 ~630 MB（98% 已满）；数据盘 `/mnt/data0` 可用 **~254 GB**

---

## 1. 引言

### 1.1 背景与目的
本项目旨在构建一个结构清晰、安全可控、可复现且具备良好泛用性的 **RSI Agent（Recursive Self-Improvement Agent，递归自我改进智能体）MVP 系统**。系统不仅支持在固定基座上做**脚手架级（Scaffold-level）自我修改**，还支持将进化探索出的高质量轨迹蒸馏回本地基座模型的 **LoRA 权重微调**，并在架构上天然适配多任务、大型数据集以及无标签（Label-Free）自进化实验。

### 1.2 核心术语定义

| 术语 | 英文 | 定义 |
| :--- | :--- | :--- |
| 递归自我改进 | Recursive Self-Improvement (RSI) | Agent 不仅改进其解决外部任务的能力，其“分析并改进自身”的逻辑（`improver`）同样处于可变区；进一步结合权重微调实现“策略 ↔ 权重”协同提升。 |
| 双轨自改进 | Dual-Track RSI | 轨道 A：Scaffold 级代码/Prompt/工具/工作流变异；轨道 B：Weight 级基于筛选轨迹的 LoRA 微调（SFT / DPO / GRPO）。 |
| 不可变底座 | Immutable Harness | 负责任务供给、沙箱隔离、验证打分、模型生命周期、版本归档与预算控制的外层系统；运行期对 Agent 严格只读/不可见。 |
| 可变本体 | Mutable Agent / Genome | 允许在迭代中被 Agent 自身修改的代码目录（推理主循环、提示词、工具函数、元改进器）。 |
| 四分数据集 | Train / Dev / Holdout / Anchor | `Train` 用于收集轨迹做 LoRA 微调；`Dev` 用于 Scaffold 反思与筛选；`Holdout` 全程隐藏验证泛化；`Anchor` 为极小金标准门禁集（尤其用于无标签进化防坍缩）。 |
| 无标签验证器 | Label-Free Verifier | 在无人工标准答案时，通过环境执行反馈、$K$ 次采样共识投票、代码-测试交叉矩阵或清单评审生成接地监督信号的模块。 |
| 模型坍缩 | Model Collapse | 多代学习自生成数据时导致的输出熵骤降、模式单一化、回音室强化错误与通用能力退化现象。 |

---

## 2. 项目范围（Scope）

### 2.1 MVP 包含范围（In-Scope）
1. **本地基座部署与微调选型**：
   - **主力基座**：`Qwen3-8B`（BF16 ~16 GB，代码/数学/工具调用全能，原生支持思考模式开关，单卡 32 GB 可部署 + LoRA 微调）。
   - **代码专项备选**：`Qwen2.5-Coder-7B-Instruct`（BF16 ~15.2 GB）。
   - **零下载烟测备选**：本机已缓存的 `DeepSeek-R1-Distill-Qwen-7B`（仅用于验证部署与微调链路）。
   - **统一推理与微调**：推理侧使用 vLLM（OpenAI 兼容 API + `--enable-lora` 动态热加载）；训练侧支持单卡 32 GB LoRA 微调（优先 `trl` / `peft`，兼容 `LLaMA-Factory`），支持**单卡分时复用**与**多卡并行**一键配置切换。
2. **轻量且能力匹配的任务与数据集**：
   - **Stage 1（极速闭环验证）**：`MBPP+`（378 题，标准库单测，快速打通系统）。
   - **Stage 2（主实验基准）**：`BigCodeBench` 分层抽样（约 240 题：Train 120 / Dev 60 / Holdout 60，7B–8B 基线 ~35%–55%，具备充足进化空间）。
   - **跨领域泛化验证任务**：`MATH-500` 抽样（约 200 题，确定性符号校验 `math-verify`/`sympy`），验证系统零改动切换任务域。
3. **泛用性与大数据集扩展能力**：
   - 插件化 `TaskAdapter` 与 `DatasetProvider`（基于 HuggingFace `datasets` Arrow/Parquet/Streaming），支持万级以上大型数据集的**两级评测漏斗（`dev_mini` 快速筛选 → `dev_full` 晋级确认）**与高并发评测。
4. **有标签与无标签统一验证体系**：
   - 支持确定性 `GroundTruthVerifier`（单测 / 符号数学）。
   - 支持无标签 `LabelFreeVerifier` 插件（`ConsensusVerifier` 多数投票共识、`CodeTestCoVerifier` 代码×自测交叉矩阵、`RubricCriticVerifier` 二值清单评审），并强制配套 `Anchor Gatekeeper` 防漂移防坍缩。
5. **自动化进化闭环与可观测性**：
   - 完整的版本谱系（同时追踪 Scaffold 快照与 LoRA Adapter 版本）、结构化 Trace、CLI 工具及自动生成 `summary.md` 实验报告。

### 2.2 MVP 排除范围（Out-of-Scope）
1. 不做多机分布式集群训练与千亿参数全量预训练。
2. 不开发 Web 前端可视化平台（MVP 仅提供 CLI 与 Markdown/JSON 产物）。
3. 沙箱默认断网，不接入开放公网搜索工具。

---

## 3. 目标用户与核心用例

- **目标用户**：AI Agent 研究者、大模型后训练/算法工程师、RSI 系统学习者。
- **核心用户故事**：
  - **US-01（单卡本地起跑）**：仅使用本机当前空闲的 1 张 RTX 5090（GPU 1），在 `/mnt/data0` 下完成模型推理服务启动与 `v0` 基线评估。
  - **US-02（双轨自我进化）**：通过配置文件启动自动进化——先执行多代 Scaffold 自改，再自动收集 Train 集验证通过的轨迹触发一轮单卡 LoRA 微调，并热加载新 LoRA 继续迭代。
  - **US-03（跨任务与大数据集切换）**：仅修改 YAML 配置即可从 `BigCodeBench` 切换到 `MATH-500` 或自定义大型 Parquet 数据集（自动启用 `dev_mini` 分层抽样漏斗）。
  - **US-04（无标签自进化实验）**：隐藏数据集标签并切换为 `ConsensusVerifier` 或 `CodeTestCoVerifier`，借助 `Anchor Gatekeeper` 监控防坍缩，并用隐藏 Holdout 量化评估无标签进化的真实收益。

---

## 4. 功能需求（Functional Requirements）

### FR-01：基线 Agent 与统一调用契约
- 提供初始种子 `agent/`（`v0`），暴露统一的 `solve(task, llm_client)` 与 `propose_improvement(context, llm_client)` 接口。
- `solve()` 需兼容多种任务域（通过 `task["domain"]` 与通用输入渲染），并支持单次推理或 $K$ 次采样。

### FR-02：插件化任务适配与大数据集供给（`TaskAdapter` & `DatasetProvider`）
- 通过注册表机制管理任务适配器（内置 `CodeUnitTestAdapter`、`MathSymbolicAdapter`、`GenericQAAdapter`）。
- 支持 `train` / `dev` / `holdout` / `anchor` 四分数据加载；支持本地 JSONL/Parquet 与 HuggingFace Datasets（Arrow 内存映射与 Streaming 模式）。
- 支持大数据集**两级评测漏斗**：日常每代在分层抽样的 `dev_mini` 上评估，仅当候选体表现达标（Top-K 或刷新纪录）时触发 `dev_full` / `holdout` 评测。

### FR-03：有标签与无标签统一验证器（`Verifier`）
- **有标签模式**：调用隔离 `pytest`（代码）或 `sympy`/`math-verify`（数学）产出确定性得分。
- **无标签模式**：
  - `ConsensusVerifier`：对单题 $K$ 次采样做等价答案聚合与多数投票，计算共识度（`consensus_ratio`）并过滤低置信样本。
  - `CodeTestCoVerifier`：对无标签代码题联合生成 $M$ 份代码与 $N$ 份测试，基于执行通过矩阵（可选变异测试）打分。
  - `RubricCriticVerifier`：基于二值检查清单与位置交换消偏执行结构化评审。

### FR-04：结构化轨迹采集与训练数据导出（Trace Store）
- 记录每次任务的完整步骤、工具调用、报错日志、Token 消耗、耗时与验证得分。
- 支持一键将 `Train` 集中验证通过（或高共识）的轨迹导出为标准微调数据格式（ShareGPT / Messages SFT 格式、Preference DPO 对、或 GRPO Prompt+Reward 数据集）。

### FR-05：元改进与安全代码修补（Meta-Agent & Patcher）
- Meta-Agent 基于父代源码、评测聚合指标与抽样失败 Trace，输出 `diagnosis`、`hypothesis`、`edits`。
- 严格执行 `agent/` 目录路径白名单、防 Symlink 逃逸、AST 语法检查与双入口契约冒烟测试。

### FR-06：隔离沙箱执行（Sandbox）
- 支持 `subprocess`（快速调试）与 `docker`（强隔离：`--network none`、内存/CPU/PID 限制、只读挂载）双后端。
- 强制单任务超时与输出截断；评测代码在全新临时目录注入不可变测试用例运行。

### FR-07：版本归档与进化选择策略（Archive & Selection）
- 同时追踪 **Scaffold 代码快照** 与 **LoRA 权重版本** 的组合谱系（`CandidateRecord`）。
- 内置 `hill_climbing` 与 `evolutionary_archive`（得分 + 多样性惩罚，类 DGM）两种父代采样策略。

### FR-08：权重级微调与模型生命周期管理（Weight Adaptation & Model Lifecycle）
- 支持周期性或条件触发（如每隔 $K$ 代或 Scaffold 收敛时）从 `Train` 集收集有效轨迹，调用 `TrainerBackend` 执行单卡 LoRA 微调（MVP 首期实现 **Rejection-Sampling LoRA SFT**，预留 DPO / GRPO 接口）。
- `ModelLifecycleManager` 自动适配硬件拓扑：
  - **单卡分时模式**：安全停止 vLLM 释放 GPU 显存 → 运行 LoRA 训练 → 重启 vLLM 并通过 `--enable-lora` 动态挂载新 Adapter。
  - **多卡并行模式**：推理卡与训练卡分离，训练完成后通过热加载 API 注册新 LoRA。

### FR-09：防坍缩护栏、锚点门禁与预算控制（Guardrails & Anchor Gatekeeper）
- **硬预算控制**：最大进化代数、最大微调轮数、总 Token / 墙钟时间上限、早停耐心值。
- **无标签专项防线**：
  - 每次权重更新或每 $M$ 代自动在 **Anchor 金标准集** 上回归校验，若 Anchor 分数下降超过阈值（如 > 3%）则拒绝合并并自动回滚。
  - 记录输出熵 / 唯一解比例与 KL 漂移指标，在报告中提示坍缩风险。

### FR-10：CLI 与实验报告
- 提供 `rsi` CLI，涵盖环境检查、服务管理、进化运行、单点评测、LoRA 微调、谱系检视、代码 Diff 与报告生成。

---

## 5. 非功能需求（Non-Functional Requirements）

| 类别 | 编号 | 具体要求 |
| :--- | :--- | :--- |
| **安全性** | NFR-01 | **严格信任边界**：可变代码无法读取 `holdout` / `anchor` 数据，无法修改 `harness/` 或历史快照。 |
| **安全性** | NFR-02 | **资源防滥用**：单任务超时 ≤ 30s，Docker 容器内存上限可配置（默认 1 GB），默认断网。 |
| **可靠性** | NFR-03 | **断点续跑与显存自愈**：每代原子落盘；主进程中断可 `--resume`；单卡模式切换前确认显存完全释放。 |
| **可复现性** | NFR-04 | 完整记录配置快照、随机种子、基座模型 Hash、LoRA Adapter 路径、数据集切分指纹。 |
| **可观测性** | NFR-05 | 全链路结构化日志（JSONL），记录推理延迟、Token、显存峰值、微调 Loss 曲线与过拟合/坍缩指标。 |
| **可扩展性** | NFR-06 | `TaskAdapter`、`DatasetProvider`、`Verifier`、`SandboxExecutor`、`TrainerBackend`、`SelectionPolicy` 全协议解耦。 |
| **存储与硬件** | NFR-07 | **强制 `/mnt/data0` 存储纪律**：严禁向系统盘 `/` 写入模型缓存、数据集、venv 或 `runs/`；默认适配单张 32 GB RTX 5090（`sm_120`），支持 YAML 配置扩至多卡。 |

---

## 6. MVP 验收标准（Definition of Done）

满足以下全部条件即视为 MVP v1.1 验收通过：

1. **存储与环境合规**：独立虚拟环境、HF 缓存与运行产物全部位于 `/mnt/data0`，系统盘 `/` 占用零增长；在单张 RTX 5090 (GPU 1) 上成功部署 `Qwen3-8B`（或烟测模型）并支持 `--enable-lora`。
2. **离线测试全绿**：Mock LLM + Mock Trainer 驱动的单元测试、集成测试、安全对抗测试（含路径穿越、死循环、无标签锚点拦截）通过率 100%。
3. **双轨闭环实测通过**：在 `MBPP+`（或 `BigCodeBench` 子集）上完成至少 5 代 Scaffold 进化 + 1 轮本地单卡 LoRA 微调与热加载评测，生成完整 `summary.md`。
4. **跨任务泛用性验证**：仅通过切换配置文件（不修改 Harness 核心代码），成功跑通 `MATH-500` 任务评测与进化流程。
5. **无标签验证闭环可运行**：在隐藏标签模式下跑通 `ConsensusVerifier` 或 `CodeTestCoVerifier`，并验证 `Anchor Gatekeeper` 能在模拟退化时正确触发拦截回滚。
