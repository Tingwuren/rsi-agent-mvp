# RSI Agent MVP

这是一个学习与实验仓库，完整实现了 **双轨递归自我改进智能体（Dual-Track Recursive Self-Improvement Agent）MVP** 系统，支持：

- **轨道 A（Scaffold RSI）**：隔离沙箱内由 Agent 自身修改 `agent/` 源码（含 `prompts.py`、`core.py`、`tools.py` 与元改进器 `improver.py`），经路径/Symlink 白名单、AST 语法与信任边界校验及双入口契约冒烟测试后进入进化筛选。
- **轨道 B（Weight Adaptation / LoRA 微调）**：从 `Train` 集筛选高置信度通过轨迹（可选混入 Replay Buffer 防模型坍缩），通过 `ModelLifecycleManager` 在单卡 32 GB GPU 上自动分时停服 → 执行独立进程 BF16 LoRA SFT → 重启/热加载新 LoRA → 经 `Anchor Gatekeeper` 防漂移门禁验收。
- **跨任务、大数据集与无标签泛用架构**：插件化 `DatasetProvider`（JSONL / Parquet Arrow 分片与 `dev_mini → dev_full` 两级漏斗）、`TaskAdapter`（`code` / `math` / `qa`）与 `Verifier`（有标签 `ground_truth_code` / `ground_truth_math` + 无标签 `consensus` / `code_test_co` + 强制金标准 `Anchor Gatekeeper`）。

---

## 快速上手

### 1. 初始化隔离环境（全部落地 `/mnt/data0`）
```bash
bash scripts/setup_env.sh --with-gpu
source .venv/bin/activate
rsi doctor --config configs/code_mbpp_plus.yaml
```

### 2. 准备标准四分数据集（`train` / `dev` / `holdout` / `anchor`）
```bash
python scripts/prepare_datasets.py --dataset all          # MBPP+ & MATH-500 & synthetic
python scripts/prepare_datasets.py --dataset bigcodebench # BigCodeBench 280 题分层子集
```

### 3. 核心 CLI 命令

| 命令 | 用途 | 示例 |
| :--- | :--- | :--- |
| `rsi doctor` | 检查 GPU 空闲状态、磁盘余量、`HF_HOME`、PyTorch `sm_120` BF16 算子 | `rsi doctor --config configs/code_mbpp_plus.yaml` |
| `rsi serve` | 在指定 GPU 启动本地 `vLLM` 服务（支持 `--enable-lora`） | `rsi serve --config configs/code_mbpp_plus.yaml --model .workspace/models/Qwen3-8B` |
| `rsi eval` | 评测种子 `v0` 或任意候选体快照（隔离沙箱） | `rsi eval --config configs/code_mbpp_plus.yaml --split dev --max-samples 32` |
| `rsi run` | 启动或 `--resume` 恢复双轨 RSI 进化闭环 | `rsi run --config configs/code_bigcodebench.yaml --max-generations 5` |
| `rsi train-lora` | 手动执行单卡 BF16 LoRA SFT 微调 | `rsi train-lora --config configs/code_mbpp_plus.yaml --train-jsonl sft.jsonl` |
| `rsi inspect` | 查看 SQLite 进化谱系表与最优候选体 | `rsi inspect --run .workspace/runs/<exp_id>` |
| `rsi diff` | 查看任意两代 `agent/` 代码 Diff 与指标变化 | `rsi diff --run .workspace/runs/<exp_id> cand_0000 cand_0003` |
| `rsi report` | 生成/刷新实验汇总报告 `summary.md`（含过拟合告警） | `rsi report --run .workspace/runs/<exp_id>` |

### 4. 一键实机实验脚本与测试
```bash
# 运行离线全量单元测试、集成测试与 SEC-01~09 安全对抗测试
ruff check src tests scripts && mypy src && pytest -v

# 单卡 GPU 1 实机运行 v0 基线（MBPP+ & MATH-500）
bash scripts/run_v0_baseline.sh

# 单卡 GPU 1 实机运行 3 代 Scaffold RSI
bash scripts/run_m3_live_rsi.sh

# 单卡 GPU 1 实机运行 6 组对照消融实验（Exp-1 ~ Exp-6）
python scripts/run_ablation_suite.py
```

---

## v1.2 实机 6 组对照消融结果（`Qwen3-8B` / `MATH-500`）

单卡 NVIDIA RTX 5090 32 GB（BF16），分层 `MATH-500`（`Train=100 / Dev=20 / Anchor=16 / Holdout=30`，Exp-6 无标注 `Train=60 / Dev=20`）：

| 实验组 | 方法 | 最优候选 (`Track`) | Dev | Anchor | **Holdout** | **Holdout 增益** | 有效变异率 | 最优 Dev Tokens |
| :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Exp-1** | Baseline v0 | `cand_0000` (`bootstrap`) | 70.0% | 62.5% | **50.0%** | +0.0% | - | 105,689 |
| **Exp-2** | Prompt-Only SI (4g) | `cand_0004` (`scaffold`) | 80.0% | 75.0% | **70.0%** | **+20.0%** | 100.0% | **68,942 (-34.8%)** |
| **Exp-3** | Pure Scaffold RSI (4g) | `cand_0002` (`scaffold`) | 80.0% | 68.8% | **73.3%** | **+23.3%** | 100.0% | 145,678 (+37.8%) |
| **Exp-4** | Pure Weight LoRA | `cand_0000` (`bootstrap`) | 70.0% | 62.5% | **50.0%** | +0.0% | - | 105,689 |
| **Exp-5** | Full Dual-Track RSI | `cand_0004` (`weight_lora/lora_0001`) | 80.0% | 75.0% | **73.3%** | **+23.3%** | 100.0% | **121,069**（较 Scaffold -19.7%） |
| **Exp-6** | Label-Free Consensus RSI | `cand_0002` (`scaffold`) | 55.0% | 62.5% | **53.3%** | **+3.3%** | 66.7% | 140,742 (-5.7%) |

---

## 工程文档

完整的软件工程设计文档（v1.1）位于 [`docs/`](./docs/README.md)：

- [01. 软件需求规格说明书 (SRS)](./docs/01-srs.md)
- [02. 系统架构设计说明书 (SAD)](./docs/02-architecture.md)
- [03. 详细设计与接口/数据契约说明书 (LLD)](./docs/03-detailed-design.md)
- [04. 测试计划与评测验证方案 (Test & Eval)](./docs/04-test-plan.md)
- [05. 研发实施计划与工程规范 (Implementation Plan)](./docs/05-implementation-plan.md)
