# RSI Agent MVP 文档中心

本目录包含 **RSI Agent MVP（Recursive Self-Improvement Agent 最小可行版本）** 的完整软件工程设计文档（**v1.1**），结合本机 **6 × RTX 5090 (32 GB, Blackwell `sm_120`)** 实测环境，覆盖需求规格、双轨自进化架构（Scaffold 级 + LoRA 权重级）、插件化多任务与大数据集抽象、有标签/无标签统一验证体系及分阶段实施计划。

## 文档索引

| 编号 | 文档名称 | 核心内容 | 版本 |
| :--- | :--- | :--- | :--- |
| 01 | [软件需求规格说明书 (SRS)](./01-srs.md) | 本机约束、项目范围（双轨 RSI + 本地部署微调 + 无标签扩展）、FR-01~10、NFR-01~07 与验收标准 | v1.1 |
| 02 | [系统架构设计说明书 (SAD)](./02-architecture.md) | 硬件拓扑与 `/mnt/data0` 存储规范、双层信任边界、双轨进化状态机、插件化任务/验证架构与目录规划 | v1.1 |
| 03 | [详细设计与接口契约 (LLD)](./03-detailed-design.md) | 通用化 Pydantic 数据模型、`TaskAdapter`/`DatasetProvider`/`Verifier`/`TrainerBackend` 协议、配置与 CLI 规范 | v1.1 |
| 04 | [测试计划与评测方案 (Test & Eval)](./04-test-plan.md) | 单元/集成/安全对抗测试矩阵、`MBPP+`/`BigCodeBench`/`MATH-500` 评测规划、无标签对照与 6 组消融实验 | v1.1 |
| 05 | [研发实施计划与工程规范 (Plan)](./05-implementation-plan.md) | M0–M5 里程碑（含 `/mnt/data0` 环境初始化与 LoRA 闭环）、WBS 任务表、代码规范与风险矩阵 | v1.1 |

## v1.1 核心演进摘要

1. **落定本机硬件与存储基线**：默认面向**单张空闲 RTX 5090（32 GB）分时运行推理 + LoRA 微调**（支持配置一键扩至多卡）；强制所有虚拟环境、模型缓存、数据集与运行产物落地 `/mnt/data0`，规避系统盘 `/` 空间耗尽风险。
2. **落定基座模型与轻量任务组合**：推荐 **`Qwen3-8B`** 为跨任务主力基座（备选 `Qwen2.5-Coder-7B-Instruct`；本机已缓存 `DeepSeek-R1-Distill-Qwen-7B` 用于零下载烟测）；任务采用 **`MBPP+`（Stage 1）→ `BigCodeBench` 抽样（Stage 2）+ `MATH-500`（跨领域泛化验证）**，升级为 **Train / Dev / Holdout / Anchor** 四分数据治理。
3. **升级双轨自我改进（Dual-Track RSI）**：在 Scaffold 级（代码/Prompt/工具/Improver 自改）之外，纳入 **Weight 级 LoRA 微调闭环**（基于 vLLM `--enable-lora` 热加载新权重）。
4. **增强泛用性与无标签进化能力**：抽象 `TaskAdapter`、`DatasetProvider`（支持 Arrow/Streaming 与两级 Mini→Full 评测漏斗）与 `Verifier` 插件（统一支持有标签客观验证与无标签共识/交叉验证），并内置 **`Anchor Gatekeeper`** 防坍缩、防漂移。
